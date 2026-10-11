/* cc1101.c: portable CC1101 driver (see cc1101.h). */
#include "cc1101.h"

#include "radio_core.h"
#include "rc_hal.h"

/* Register table: termoweb_rx 3.5's cc1101.c CONFIG[] (and the ESP32's),
 * minus FREQ2/1/0 and SYNC1/0, which come from runtime state. The values
 * not tied to a measured link parameter are the generic 2-FSK baseline of
 * the ELECHOUSE/SmartRC CC1101 driver, cross-checked with the datasheet. */
static const uint8_t CONFIG[][2] RC_PROGMEM = {
    {CC1101_IOCFG2, 0x00},     /* GDO2: RX FIFO at/above threshold -> rc_on_gdo2 */
    {CC1101_IOCFG0, 0x06},     /* GDO0: asserts on sync, deasserts at packet end -> rc_on_gdo0 */
    {CC1101_FIFOTHR, 0x40},    /* ADC_RETENTION + 4-byte RX threshold: fastest read of byte 0 */
    {CC1101_PKTLEN, RC_PKT_LEN}, /* don't-care in infinite mode; rewritten per frame */
    {CC1101_PKTCTRL1, 0x04},   /* APPEND_STATUS: RSSI + LQI/CRC_OK follow each packet */
    {CC1101_PKTCTRL0, 0x02},   /* infinite length until byte 0 gives the total; CRC, whitening off */
    {CC1101_ADDR, 0x00},
    {CC1101_CHANNR, 0x00},
    {CC1101_FSCTRL1, 0x06},
    {CC1101_FSCTRL0, 0x00},
    {CC1101_MDMCFG4, 0x68},    /* 270.8 kHz RX BW, DRATE_E=8 */
    {CC1101_MDMCFG3, 0x83},    /* 9596 bps */
    {CC1101_MDMCFG2, 0x02},    /* 2-FSK, 16/16 sync */
    {CC1101_MDMCFG1, 0x42},    /* 8-byte TX preamble */
    {CC1101_MDMCFG0, 0xF8},
    {CC1101_DEVIATN, 0x50},    /* 50.8 kHz deviation */
    {CC1101_MCSM1, 0x30},      /* IDLE after packet; the firmware re-arms RX itself */
    {CC1101_MCSM0, 0x18},      /* autocal on IDLE->RX, so every SRX (and 'F') recalibrates */
    {CC1101_FOCCFG, 0x16},
    {CC1101_BSCFG, 0x1C},
    {CC1101_AGCCTRL2, 0xC7},
    {CC1101_AGCCTRL1, 0x00},
    {CC1101_AGCCTRL0, 0xB2},
    {CC1101_FREND1, 0x56},
    {CC1101_FREND0, 0x11},     /* PA index 1: PATABLE[0..1], both RC_PA */
    {CC1101_FSCAL3, 0xE9},
    {CC1101_FSCAL2, 0x2A},
    {CC1101_FSCAL1, 0x00},
    {CC1101_FSCAL0, 0x1F},
    {CC1101_FSTEST, 0x59},
    {CC1101_TEST2, 0x81},      /* required for RX BW <= 325 kHz */
    {CC1101_TEST1, 0x35},
    {CC1101_TEST0, 0x09},
};

/* TXBYTES reads back 0 when the FIFO is exactly full (silicon erratum), so
 * fills stop one byte short of 64. */
#define TX_FIFO_SIZE 64
#define TX_INITIAL_FILL 63
#define TX_REFILL_THRESHOLD 32
#define TX_POLLS 2000

void cc1101_write_freq(uint32_t khz) {
  uint32_t word = rc_freq_word(khz);
  rc_hal_write_reg(CC1101_FREQ2, (uint8_t)(word >> 16));
  rc_hal_write_reg(CC1101_FREQ1, (uint8_t)(word >> 8));
  rc_hal_write_reg(CC1101_FREQ0, (uint8_t)word);
}

void cc1101_write_sync(void) {
  rc_hal_write_reg(CC1101_SYNC1, RC_SYNC1);
  rc_hal_write_reg(CC1101_SYNC0, rc.dialect == RC_DIALECT_B ? RC_SYNC0_B : RC_SYNC0_A);
}

void cc1101_init(void) {
  rc_hal_chip_wake();
  rc_hal_strobe(CC1101_SRES);
  /* SRES restarts the crystal; the next transfer waits for MISO low. */
  rc_hal_delay_us(2000);
  for (uint8_t i = 0; i < sizeof(CONFIG) / sizeof(CONFIG[0]); i++)
    rc_hal_write_reg(RC_RD(&CONFIG[i][0]), RC_RD(&CONFIG[i][1]));
  cc1101_write_freq(rc.freq_khz);
  cc1101_write_sync();
  rc_hal_strobe(CC1101_SCAL);
  rc_hal_delay_us(1000);
  uint8_t pa[2] = {RC_PA, RC_PA};
  rc_hal_write_burst(CC1101_PATABLE, pa, 2);
}

void cc1101_enter_rx(void) {
  rc_hal_strobe(CC1101_SIDLE);
  rc_hal_strobe(CC1101_SFRX);
  rc_hal_write_reg(CC1101_PKTCTRL0, 0x02);
  rc_hal_strobe(CC1101_SRX);
}

void cc1101_set_fixed_length(uint8_t total) {
  rc_hal_write_reg(CC1101_PKTLEN, total);
  rc_hal_write_reg(CC1101_PKTCTRL0, 0x00);
}

void cc1101_enter_async_rx(void) {
  rc_hal_strobe(CC1101_SIDLE);
  rc_hal_write_reg(CC1101_IOCFG0, 0x0D);   /* async serial data out */
  rc_hal_write_reg(CC1101_IOCFG2, 0x0E);   /* carrier sense (diagnostic) */
  rc_hal_write_reg(CC1101_PKTCTRL0, 0x32); /* async serial, infinite */
  rc_hal_strobe(CC1101_SRX);
}

static uint8_t tx_recover(void) {
  rc_hal_strobe(CC1101_SFTX);
  cc1101_enter_rx();
  return CC1101_TX_UNDERFLOW;
}

/* len <= 64 goes in one burst before STX; longer frames start on a 63-byte
 * fill and are topped up whenever TXBYTES drops under 32. Every wait is
 * bounded (2000 polls of 100 us). */
uint8_t cc1101_transmit(const uint8_t *buf, uint8_t len) {
  rc_hal_strobe(CC1101_SIDLE);
  rc_hal_strobe(CC1101_SFTX);
  /* RX may be in infinite length mode; TX needs fixed length. */
  rc_hal_write_reg(CC1101_PKTCTRL0, 0x00);
  rc_hal_write_reg(CC1101_PKTLEN, len);

  if (len <= TX_FIFO_SIZE) {
    rc_hal_write_burst(CC1101_FIFO_ADDR, buf, len);
    rc_hal_strobe(CC1101_STX);
  } else {
    rc_hal_write_burst(CC1101_FIFO_ADDR, buf, TX_INITIAL_FILL);
    rc_hal_strobe(CC1101_STX);
    uint8_t sent = TX_INITIAL_FILL;
    for (uint16_t poll = 0; sent < len; poll++) {
      if (poll >= TX_POLLS) {
        uint8_t stuck = rc_hal_read_status(CC1101_MARCSTATE) & 0x1F;
        rc_hal_strobe(CC1101_SFTX);
        cc1101_enter_rx();
        return stuck;
      }
      uint8_t txbytes = rc_hal_read_status(CC1101_TXBYTES);
      if (txbytes & 0x80) return tx_recover();
      uint8_t occupied = txbytes & 0x7F;
      if (occupied < TX_REFILL_THRESHOLD) {
        uint8_t space = (uint8_t)(TX_FIFO_SIZE - occupied - 1);
        uint8_t remaining = (uint8_t)(len - sent);
        uint8_t chunk = remaining < space ? remaining : space;
        if (chunk > 0) {
          rc_hal_write_burst(CC1101_FIFO_ADDR, buf + sent, chunk);
          sent = (uint8_t)(sent + chunk);
        }
      }
      rc_hal_delay_us(100);
    }
  }

  uint8_t state = 0;
  for (uint16_t i = 0; i < TX_POLLS; i++) {
    rc_hal_delay_us(100);
    state = rc_hal_read_status(CC1101_MARCSTATE) & 0x1F;
    if (state == CC1101_MARCSTATE_IDLE) break;
    if (state == CC1101_MARCSTATE_TXFIFO_UNDERFLOW) return tx_recover();
  }
  cc1101_enter_rx();
  return state == CC1101_MARCSTATE_IDLE ? 0 : state;
}

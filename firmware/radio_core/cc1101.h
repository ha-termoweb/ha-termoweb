/* cc1101.h: portable CC1101 driver for the TermoWeb 869 MHz link.
 *
 * Built on the register-level SPI functions in rc_hal.h, so the same code
 * runs on the AVR nanoCUL and (later) the ESP32. Ported from the nanoCUL
 * termoweb_rx 3.5 cc1101.c and esphome/termoweb_radio.h; the register table
 * and the transmit/refill logic are unchanged. FREQ and SYNC are written
 * from runtime state (the 'F' and 'Y' commands), not from the table.
 */
#ifndef RC_CC1101_H
#define RC_CC1101_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Config registers (CC1101 datasheet Table 45). */
#define CC1101_IOCFG2 0x00
#define CC1101_IOCFG0 0x02
#define CC1101_FIFOTHR 0x03
#define CC1101_SYNC1 0x04
#define CC1101_SYNC0 0x05
#define CC1101_PKTLEN 0x06
#define CC1101_PKTCTRL1 0x07
#define CC1101_PKTCTRL0 0x08
#define CC1101_ADDR 0x09
#define CC1101_CHANNR 0x0A
#define CC1101_FSCTRL1 0x0B
#define CC1101_FSCTRL0 0x0C
#define CC1101_FREQ2 0x0D
#define CC1101_FREQ1 0x0E
#define CC1101_FREQ0 0x0F
#define CC1101_MDMCFG4 0x10
#define CC1101_MDMCFG3 0x11
#define CC1101_MDMCFG2 0x12
#define CC1101_MDMCFG1 0x13
#define CC1101_MDMCFG0 0x14
#define CC1101_DEVIATN 0x15
#define CC1101_MCSM1 0x17
#define CC1101_MCSM0 0x18
#define CC1101_FOCCFG 0x19
#define CC1101_BSCFG 0x1A
#define CC1101_AGCCTRL2 0x1B
#define CC1101_AGCCTRL1 0x1C
#define CC1101_AGCCTRL0 0x1D
#define CC1101_FREND1 0x21
#define CC1101_FREND0 0x22
#define CC1101_FSCAL3 0x23
#define CC1101_FSCAL2 0x24
#define CC1101_FSCAL1 0x25
#define CC1101_FSCAL0 0x26
#define CC1101_FSTEST 0x29
#define CC1101_TEST2 0x2C
#define CC1101_TEST1 0x2D
#define CC1101_TEST0 0x2E

/* Status registers: read one at a time with the burst bit set (a status
 * "burst" re-reads the same register, so they cannot be batched). */
#define CC1101_PARTNUM 0x30
#define CC1101_VERSION 0x31
#define CC1101_RSSI 0x34
#define CC1101_MARCSTATE 0x35
#define CC1101_PKTSTATUS 0x38
#define CC1101_TXBYTES 0x3A
#define CC1101_RXBYTES 0x3B

/* Command strobes. */
#define CC1101_SRES 0x30
#define CC1101_SCAL 0x33
#define CC1101_SRX 0x34
#define CC1101_STX 0x35
#define CC1101_SIDLE 0x36
#define CC1101_SFRX 0x3A
#define CC1101_SFTX 0x3B
#define CC1101_PATABLE 0x3E

/* SPI header bits. */
#define CC1101_WRITE_BURST 0x40
#define CC1101_READ_BURST 0xC0
#define CC1101_FIFO_ADDR 0x3F

/* cc1101_transmit's return for a TX FIFO underflow: outside the 5-bit
 * MARCSTATE range, so it is never confused with "stuck in MARCSTATE 0x16". */
#define CC1101_TX_UNDERFLOW 0xFF
#define CC1101_MARCSTATE_IDLE 0x01
#define CC1101_MARCSTATE_TXFIFO_UNDERFLOW 0x16

/* Is a CC1101 on the bus? Reads PARTNUM and VERSION (a CC1101 reports
 * 00/14, older silicon 00/04). An empty socket reads all 0x00 or all 0xFF,
 * whatever MISO idles at, so VERSION 0x00 or 0xFF (or PARTNUM 0xFF) means
 * no chip. Unknown non-zero versions (clones) count as present. */
uint8_t cc1101_present(void);
/* SRES, the register table, FREQ (rc.freq_khz), SYNC (rc.dialect), SCAL,
 * PATABLE (RC_PA). */
void cc1101_init(void);
/* Flush the RX FIFO and re-arm RX in infinite length mode: the next frame's
 * length is unknown until its byte 0 arrives. */
void cc1101_enter_rx(void);
/* Switch to fixed length mid-reception; the chip's own byte counter keeps
 * running, so the packet ends exactly at `total` and GDO0 falls there. */
void cc1101_set_fixed_length(uint8_t total);
void cc1101_write_freq(uint32_t khz);
void cc1101_write_sync(void);
/* Send len bytes as one fixed-length packet (the chip adds preamble and
 * sync), refilling the 64-byte TX FIFO for longer frames, then re-arm RX.
 * Returns 0, CC1101_TX_UNDERFLOW, or the MARCSTATE that never reached IDLE. */
uint8_t cc1101_transmit(const uint8_t *buf, uint8_t len);
/* Asynchronous serial mode for the raw survey: GDO0 = demodulator output,
 * GDO2 = carrier sense, no sync word or framing. cc1101_init() undoes it. */
void cc1101_enter_async_rx(void);

#ifdef __cplusplus
}
#endif

#endif

/* Host HAL for testing firmware/radio_core with ctypes: a fake CC1101.
 *
 * Output bytes collect in a buffer; the clock is a counter (delays advance
 * it); the chip keeps config registers, an RX FIFO the test fills, and the
 * bytes each transmit wrote to the TX FIFO. A transmit always "finishes"
 * (MARCSTATE reads IDLE) unless the test sets a stuck MARCSTATE.
 */
#include <stdint.h>
#include <string.h>

#include "cc1101.h"
#include "radio_core.h"
#include "rc_hal.h"

static char out[1 << 16];
static uint32_t out_len;
static uint32_t clock_us;
static uint8_t regs[0x40];
static uint8_t status_regs[0x40];
static uint8_t rx_fifo[4096];
static uint32_t rx_head, rx_tail;
static uint8_t rx_overflow;
static uint8_t tx_frame[512];
static uint32_t tx_len;
static uint32_t tx_count;
static uint8_t marcstate_tx = CC1101_MARCSTATE_IDLE;
static uint32_t strobes[0x40];
static int masked;
static int lock_depth;
static int survey_calls;
static int last_survey_secs = -1;

void rc_hal_putc(char c) {
  if (out_len < sizeof(out) - 1) out[out_len++] = c;
}
uint32_t rc_hal_micros(void) { return clock_us; }
void rc_hal_delay_us(uint16_t us) { clock_us += us; }
void rc_hal_chip_wake(void) {}

uint8_t rc_hal_strobe(uint8_t cmd) {
  strobes[cmd & 0x3F]++;
  if (cmd == CC1101_SFRX) rx_head = rx_tail = rx_overflow = 0;
  if (cmd == CC1101_SFTX) tx_len = 0;
  if (cmd == CC1101_STX) tx_count++;
  return 0;
}

void rc_hal_write_reg(uint8_t addr, uint8_t val) { regs[addr & 0x3F] = val; }

void rc_hal_write_burst(uint8_t addr, const uint8_t *buf, uint8_t len) {
  if (addr == CC1101_FIFO_ADDR) {
    for (uint8_t i = 0; i < len && tx_len < sizeof(tx_frame); i++) tx_frame[tx_len++] = buf[i];
  } else if (addr == CC1101_PATABLE) {
    regs[CC1101_PATABLE] = buf[0];
  }
}

uint8_t rc_hal_read_status(uint8_t addr) {
  switch (addr) {
    case CC1101_RXBYTES: {
      uint32_t n = rx_tail - rx_head;
      return (uint8_t)((n > 64 ? 64 : n) | (rx_overflow ? 0x80 : 0));
    }
    case CC1101_TXBYTES:
      return 0;
    case CC1101_MARCSTATE:
      return marcstate_tx;
    default:
      return status_regs[addr & 0x3F];
  }
}

void rc_hal_read_burst(uint8_t addr, uint8_t *buf, uint8_t len) {
  (void)addr;
  for (uint8_t i = 0; i < len; i++) buf[i] = rx_head < rx_tail ? rx_fifo[rx_head++] : 0;
}

void rc_hal_gdo_mask(void) { masked++; }
void rc_hal_gdo_unmask(void) { masked--; }
void rc_hal_lock(void) { lock_depth++; }
void rc_hal_unlock(void) { lock_depth--; }

void rc_hal_survey(uint16_t secs) {
  survey_calls++;
  last_survey_secs = secs;
  if (secs) {
    rc_hal_gdo_mask(); /* a port masks packet edges while it surveys */
    rc.survey_active = 1;
    rc_print_survey_on(secs, -1000, -920);
  } else {
    rc.survey_active = 0;
    rc_survey_finish(0);
  }
}

/* ---- test controls ---- */

void host_reset(void) {
  memset(&rc, 0, sizeof(rc));
  out_len = 0;
  clock_us = 1000;
  memset(regs, 0, sizeof(regs));
  memset(status_regs, 0, sizeof(status_regs));
  memset(strobes, 0, sizeof(strobes));
  rx_head = rx_tail = rx_overflow = 0;
  tx_len = tx_count = 0;
  marcstate_tx = CC1101_MARCSTATE_IDLE;
  masked = lock_depth = survey_calls = 0;
  last_survey_secs = -1;
}

uint32_t host_take_output(char *buf, uint32_t cap) {
  uint32_t n = out_len < cap ? out_len : cap;
  memcpy(buf, out, n);
  out_len = 0;
  return n;
}

void host_advance(uint32_t us) { clock_us += us; }
void host_rx_push(const uint8_t *data, uint32_t len) {
  for (uint32_t i = 0; i < len && rx_tail < sizeof(rx_fifo); i++) rx_fifo[rx_tail++] = data[i];
}
void host_set_rx_overflow(uint8_t on) { rx_overflow = on; }
void host_set_status(uint8_t addr, uint8_t val) { status_regs[addr & 0x3F] = val; }
void host_set_marcstate_tx(uint8_t val) { marcstate_tx = val; }
uint8_t host_reg(uint8_t addr) { return regs[addr & 0x3F]; }
uint32_t host_strobes(uint8_t cmd) { return strobes[cmd & 0x3F]; }
uint32_t host_tx_count(void) { return tx_count; }
uint32_t host_tx_take(uint8_t *buf) {
  memcpy(buf, tx_frame, tx_len);
  return tx_len;
}
int host_masked(void) { return masked; }
int host_lock_depth(void) { return lock_depth; }
int host_survey_calls(void) { return survey_calls; }
int host_last_survey_secs(void) { return last_survey_secs; }
uint8_t host_dialect(void) { return rc.dialect; }
uint8_t host_auto_ack(void) { return rc.auto_ack; }
uint8_t host_our_id(void) { return rc.our_id; }
uint16_t host_net(void) { return rc.net; }
uint8_t host_rx_state(void) { return rc.rx_state; }
uint8_t host_pstate(void) { return rc.pstate; }
uint32_t host_frames_ok(void) { return rc.frames_ok; }
uint32_t host_freq_khz(void) { return rc.freq_khz; }
void host_set_mac(const char *mac) { rc.mac = mac; }
void host_set_survey_active(uint8_t on) { rc.survey_active = on; }
void host_clear_packet_ready(void) { rc.packet_ready = 0; }

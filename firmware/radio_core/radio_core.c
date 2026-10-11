/* radio_core.c: dialects, command parser, reply lines and the receive
 * state machine shared by the termoweb_rx ports (see radio_core.h).
 *
 * Behaviour is that of esphome/termoweb_radio.h 3.7-esp32, function for
 * function (its comments explain the history; termoweb_rx 3.5's main.c is
 * where the receive logic started). Where that file and this one disagree,
 * that is a bug here.
 */
#include "radio_core.h"

#include <string.h>

#include "cc1101.h"
#include "rc_hal.h"

struct rc_state rc;

/* Keeps the compiler from moving buffer reads above a volatile flag read. */
#define RC_BARRIER() __asm__ __volatile__("" ::: "memory")

/* ------------------------------------------------------------ output */

static void put_p(const char *s) {
  char c;
  while ((c = (char)RC_RD(s++)) != '\0') rc_hal_putc(c);
}

static void put_ram(const char *s) {
  while (*s) rc_hal_putc(*s++);
}

void rc_print_eol(void) {
  rc_hal_putc('\r');
  rc_hal_putc('\n');
}

static void put_line_p(const char *s) {
  put_p(s);
  rc_print_eol();
}

static void put_nibble(uint8_t v) { rc_hal_putc((char)(v < 10 ? '0' + v : 'A' + v - 10)); }

static void put_hex8(uint8_t v) {
  put_nibble(v >> 4);
  put_nibble(v & 0x0F);
}

static void put_hex(const uint8_t *b, uint16_t n) {
  for (uint16_t i = 0; i < n; i++) put_hex8(b[i]);
}

static void put_u32(uint32_t v) {
  char buf[10];
  uint8_t i = 0;
  do {
    buf[i++] = (char)('0' + v % 10);
    v /= 10;
  } while (v);
  while (i) rc_hal_putc(buf[--i]);
}

static void put_dbm10(int16_t d) {
  if (d < 0) {
    rc_hal_putc('-');
    d = (int16_t)-d;
  }
  put_u32((uint32_t)(d / 10));
  rc_hal_putc('.');
  rc_hal_putc((char)('0' + d % 10));
}

/* "869.525" */
static void put_freq(uint32_t khz) {
  uint16_t frac = (uint16_t)(khz % 1000);
  put_u32(khz / 1000);
  rc_hal_putc('.');
  rc_hal_putc((char)('0' + frac / 100));
  rc_hal_putc((char)('0' + (frac / 10) % 10));
  rc_hal_putc((char)('0' + frac % 10));
}

static uint8_t sync0(void) { return rc.dialect == RC_DIALECT_B ? RC_SYNC0_B : RC_SYNC0_A; }

static void put_sync(void) {
  put_hex8(RC_SYNC1);
  put_hex8(sync0());
}

static void put_mac(void) {
  if (rc.mac == 0) return;
  put_p(RC_PSTR(" mac="));
  put_ram(rc.mac);
}

/* Banner and Q: termoweb_rx's lines field for field; newer fields are only
 * ever appended, so parsers of the old lines still match. */
void rc_print_banner(void) {
  put_p(RC_PSTR("# termoweb_rx " RC_VERSION " freq="));
  put_freq(rc.freq_khz);
  put_p(RC_PSTR(" rate=9.6k sync="));
  put_sync();
  put_p(RC_PSTR(" mode=dynamic tx=pa"));
  put_hex8(RC_PA);
  put_mac();
  if (!rc.radio_ok) put_p(RC_PSTR(" radio=missing"));
  rc_print_eol();
}

void rc_print_query(void) {
  put_p(RC_PSTR("# Q termoweb_rx " RC_VERSION " freq="));
  put_freq(rc.freq_khz);
  put_p(RC_PSTR(" pa="));
  put_hex8(RC_PA);
  put_p(RC_PSTR(" sync="));
  put_sync();
  put_p(rc.auto_ack ? RC_PSTR(" mode=dynamic autoack=on id=") : RC_PSTR(" mode=dynamic autoack=off id="));
  put_hex8(rc.our_id);
  put_p(rc.dialect == RC_DIALECT_B ? RC_PSTR(" dialect=B net=") : RC_PSTR(" dialect=A net="));
  put_hex8((uint8_t)(rc.net >> 8));
  put_hex8((uint8_t)rc.net);
  put_mac();
  put_p(rc.radio_ok ? RC_PSTR(" radio=ok") : RC_PSTR(" radio=missing"));
  rc_print_eol();
}

void rc_print_no_radio(void) { put_line_p(RC_PSTR("# error: no CC1101 radio found")); }

void rc_print_overrun(void) { put_line_p(RC_PSTR("# uart overrun")); }

void rc_print_self_test_failed(void) {
  put_line_p(RC_PSTR("# self-test FAILED: auto-acks and crc_ok verdicts are not trustworthy"));
}

void rc_print_survey_on(uint16_t secs, int16_t floor_dbm10, int16_t thr_dbm10) {
  put_p(RC_PSTR("# survey on secs="));
  put_u32(secs);
  put_p(RC_PSTR(" floor="));
  put_dbm10(floor_dbm10);
  put_p(RC_PSTR(" thr="));
  put_dbm10(thr_dbm10);
  rc_print_eol();
}

void rc_print_survey_off(uint16_t bursts) {
  put_p(RC_PSTR("# survey off bursts="));
  put_u32(bursts);
  rc_print_eol();
}

void rc_print_rawb(uint16_t burst, int16_t peak_dbm10) {
  put_p(RC_PSTR("RAWB "));
  put_u32(burst);
  rc_hal_putc(' ');
  put_dbm10(peak_dbm10);
  rc_print_eol();
}

void rc_print_raw_start(uint16_t burst) {
  put_p(RC_PSTR("RAW "));
  put_u32(burst);
}

void rc_print_raw_run(uint8_t level, uint16_t us) {
  rc_hal_putc(' ');
  rc_hal_putc(level ? '+' : '-');
  put_u32(us);
}

void rc_print_rawe(uint16_t burst, uint16_t runs) {
  put_p(RC_PSTR("RAWE "));
  put_u32(burst);
  rc_hal_putc(' ');
  put_u32(runs);
  rc_print_eol();
}

/* ----------------------------------------------------------- protocol */

static uint16_t ccitt_step(uint16_t crc, uint8_t byte) {
  crc ^= (uint16_t)byte << 8;
  for (uint8_t b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  return crc;
}

uint16_t rc_crc_ccitt(const uint8_t *data, uint16_t len) {
  uint16_t crc = 0x1D0F;
  for (uint16_t i = 0; i < len; i++) crc = ccitt_step(crc, data[i]);
  return crc ^ 0xFFFF;
}

uint16_t rc_crc_modbus(const uint8_t *data, uint16_t len) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) crc = (crc & 1) ? (uint16_t)((crc >> 1) ^ 0xA001) : (uint16_t)(crc >> 1);
  }
  return crc;
}

/* The 8-bit register formulation of dialect.keystream(): its first bytes
 * are FF 87 B8 59 B7 A1 CC 24. */
void rc_pn9_init(struct rc_pn9 *p) {
  p->reg = 0xFF;
  p->fb = 1;
}

uint8_t rc_pn9_next(struct rc_pn9 *p) {
  uint8_t out = p->reg;
  for (uint8_t k = 0; k < 8; k++) {
    uint8_t fb_old = p->fb;
    p->fb = (uint8_t)(((p->reg ^ (uint8_t)(p->reg << 5)) >> 7) & 1);
    p->reg = (uint8_t)((p->reg << 1) | fb_old);
  }
  return out;
}

/* A: byte 0 is (total - 3) scrambled by the keystream's first byte, 0xFF.
 * B: byte 0 is the total, in the clear. */
uint16_t rc_frame_total(uint8_t dialect, uint8_t byte0) {
  uint16_t total = dialect == RC_DIALECT_B ? byte0 : (uint16_t)((uint8_t)(byte0 ^ 0xFF) + 3);
  return (total < RC_RX_TOTAL_MIN || total > RC_RX_TOTAL_MAX) ? 0 : total;
}

/* Only the verdict is computed; the frame is never altered (the host gets
 * the bytes as received). Dialect A descrambles on the fly, because its CRC
 * is over the logical bytes. */
uint8_t rc_frame_crc_ok(uint8_t dialect, const uint8_t *air, uint16_t total) {
  if (total < 5 || total > RC_RX_TOTAL_MAX) return 0;
  uint16_t crc;
  uint8_t hi, lo;
  if (dialect == RC_DIALECT_B) {
    if (air[0] != total) return 0;
    crc = rc_crc_modbus(air, (uint16_t)(total - 2));
    hi = air[total - 2];
    lo = air[total - 1];
  } else {
    struct rc_pn9 p;
    rc_pn9_init(&p);
    crc = 0x1D0F;
    for (uint16_t i = 0; i < total - 2; i++) {
      uint8_t b = air[i] ^ rc_pn9_next(&p);
      if (i == 0 && b != total - 3) return 0;
      crc = ccitt_step(crc, b);
    }
    crc ^= 0xFFFF;
    hi = air[total - 2] ^ rc_pn9_next(&p);
    lo = air[total - 1] ^ rc_pn9_next(&p);
  }
  return hi == (uint8_t)(crc >> 8) && lo == (uint8_t)crc;
}

uint8_t rc_field(uint8_t dialect, const uint8_t *air, uint8_t i) {
  if (dialect == RC_DIALECT_B) return air[i];
  struct rc_pn9 p;
  rc_pn9_init(&p);
  uint8_t ks = 0;
  for (uint8_t k = 0; k <= i; k++) ks = rc_pn9_next(&p);
  return air[i] ^ ks;
}

/* A: logical 05 <net> <acker> <acked sender> 80, CCITT CRC, scrambled.
 * B: 08 <net> <acker> <acked sender> 80, MODBUS CRC, in the clear. Built
 * here, not by the host, because a heater's ack window (~16 ms) is far
 * shorter than a host round trip. */
void rc_build_ack(uint8_t dialect, uint16_t net, uint8_t acker, uint8_t src, uint8_t out[RC_ACK_LEN]) {
  out[0] = dialect == RC_DIALECT_B ? 0x08 : 0x05;
  out[1] = (uint8_t)(net >> 8);
  out[2] = (uint8_t)net;
  out[3] = acker;
  out[4] = src;
  out[5] = 0x80;
  uint16_t crc = dialect == RC_DIALECT_B ? rc_crc_modbus(out, 6) : rc_crc_ccitt(out, 6);
  out[6] = (uint8_t)(crc >> 8);
  out[7] = (uint8_t)crc;
  if (dialect != RC_DIALECT_B) {
    struct rc_pn9 p;
    rc_pn9_init(&p);
    for (uint8_t i = 0; i < RC_ACK_LEN; i++) out[i] ^= rc_pn9_next(&p);
  }
}

/* round(kHz * 2^16 / 26000) == floor((kHz * 4096 + 812) / 1625), exact and
 * within 32 bits for kHz <= 1048000 (no 64-bit division on the AVR). */
uint32_t rc_freq_word(uint32_t khz) { return (khz * 4096UL + 812UL) / 1625UL; }

int16_t rc_rssi_dbm10(uint8_t raw) { return (int16_t)((int16_t)(int8_t)raw * 5 - 740); }

/* The ESP32's boot self-test vectors (computed in Python from the host
 * codec, synthetic net 1234): ack 01 -> 06 in A and B; a B
 * registration frame and an A poll pass in their own dialect only. */
static const uint8_t ST_ACK_A[8] RC_PROGMEM = {0xFA, 0x95, 0x8C, 0x58, 0xB1, 0x21, 0xEB, 0x3B};
static const uint8_t ST_ACK_B[8] RC_PROGMEM = {0x08, 0x12, 0x34, 0x01, 0x06, 0x80, 0xA0, 0xE4};
static const uint8_t ST_REG_B[15] RC_PROGMEM = {0x0F, 0x12, 0x34, 0x06, 0x01, 0x00, 0x06, 0x01,
                                                0x01, 0x01, 0x01, 0x00, 0x50, 0x72, 0x22};
static const uint8_t ST_POLL_A[16] RC_PROGMEM = {0xF2, 0x95, 0x8C, 0x58, 0xB1, 0xA1, 0xCD, 0x22,
                                                 0x57, 0x5E, 0x4B, 0x9C, 0x59, 0xBC, 0x0A, 0x24};

static void copy_p(uint8_t *dst, const uint8_t *src, uint8_t n) {
  for (uint8_t i = 0; i < n; i++) dst[i] = RC_RD(src + i);
}

uint8_t rc_self_test(void) {
  uint8_t want[16], got[8];
  uint8_t ok = 1;
  copy_p(want, ST_ACK_A, 8);
  rc_build_ack(RC_DIALECT_A, 0x1234, 0x01, 0x06, got);
  ok &= memcmp(got, want, 8) == 0;
  ok &= rc_frame_crc_ok(RC_DIALECT_A, want, 8);
  copy_p(want, ST_ACK_B, 8);
  rc_build_ack(RC_DIALECT_B, 0x1234, 0x01, 0x06, got);
  ok &= memcmp(got, want, 8) == 0;
  ok &= rc_frame_crc_ok(RC_DIALECT_B, want, 8);
  copy_p(want, ST_POLL_A, 16);
  ok &= rc_frame_crc_ok(RC_DIALECT_A, want, 16) && !rc_frame_crc_ok(RC_DIALECT_B, want, 16);
  copy_p(want, ST_REG_B, 15);
  ok &= rc_frame_crc_ok(RC_DIALECT_B, want, 15) && !rc_frame_crc_ok(RC_DIALECT_A, want, 15);
  return ok;
}

/* ------------------------------------------------------- receive path */

static void rx_restart(void) {
  rc_hal_strobe(CC1101_SIDLE);
  rc_hal_strobe(CC1101_SFRX);
  cc1101_enter_rx();
  rc.rx_have = 0;
  rc.rx_state = RC_RX_IDLE;
}

/* Copy whatever the RX FIFO holds into rx_buf; once byte 0 is in, compute
 * the total from it and switch the chip to fixed length mid-reception. A
 * FIFO overflow, or a total outside [8, 255], flushes and re-arms. */
static void rx_drain_fifo(void) {
  uint8_t status = rc_hal_read_status(CC1101_RXBYTES);
  if (status & 0x80) {
    rx_restart();
    return;
  }
  uint8_t n = status & 0x7F;
  if (n == 0) return;
  uint16_t have = rc.rx_have;
  uint16_t room = (uint16_t)(RC_RX_BUF_LEN - have);
  uint8_t to_read = (n > room) ? (uint8_t)room : n;
  if (to_read > 0) {
    rc_hal_read_burst(CC1101_FIFO_ADDR, &RC_RX_BUF[have], to_read);
    rc.rx_have = (uint16_t)(have + to_read);
  }
  if (rc.rx_state == RC_RX_LENPENDING && rc.rx_have >= 1) {
    uint16_t total = rc_frame_total(rc.dialect, RC_RX_BUF[0]);
    if (total == 0) {
      rx_restart();
      return;
    }
    rc.rx_total = total;
    cc1101_set_fixed_length((uint8_t)total);
    rc.rx_state = RC_RX_FIXED;
  }
}

/* GDO2 rose: RX FIFO at/above the 4-byte threshold. Repeats through a
 * frame, draining it in small chunks so a frame past 64 bytes never
 * overflows the FIFO. Ignored while RX_DONE: the previous frame still owns
 * rx_buf and new bytes wait in the hardware FIFO. */
void rc_on_gdo2(void) {
  if (rc.rx_state == RC_RX_DONE) return;
  if (rc.rx_state == RC_RX_IDLE) {
    rc.rx_have = 0;
    rc.rx_state = RC_RX_LENPENDING;
  }
  rx_drain_fifo();
}

/* GDO0 fell: end of packet once PKTLEN is in. Collect the tail (with the
 * 2 APPEND_STATUS bytes), hand the frame to rc_service, re-arm RX. */
void rc_on_gdo0(uint32_t edge_us) {
  rc.sync_count++;
  if (rc.rx_state == RC_RX_FIXED || rc.rx_state == RC_RX_LENPENDING) {
    rx_drain_fifo();
    rc.rx_done_micros = edge_us;
    rc.rx_state = RC_RX_DONE;
    rc.packet_ready = 1;
  }
  cc1101_enter_rx();
}

/* Ack first, straight from rx_buf, so it goes on air before any output;
 * then snapshot, release the receiver, catch up on the FIFO, and only then
 * print. The RX line's crc_ok is the dialect's real CRC verdict, and
 * auto-ack only answers a CRC-valid frame addressed to us that is not
 * itself an ack (flags 0x00). */
static void handle_packet(void) {
  rc.packet_ready = 0;
  RC_BARRIER();
  uint16_t total = rc.rx_total;
  uint16_t have = rc.rx_have;
  uint32_t t = rc.rx_done_micros;
  uint8_t d = rc.dialect;
  uint8_t ack_sent = 0;
  uint8_t ack[RC_ACK_LEN];

  uint8_t crc_ok = have >= total && rc_frame_crc_ok(d, RC_RX_BUF, total);
  if (crc_ok) rc.frames_ok++;

  if (rc.auto_ack && crc_ok) {
    uint8_t src = rc_field(d, RC_RX_BUF, 3);
    uint8_t dst = rc_field(d, RC_RX_BUF, 4);
    uint8_t flags = rc_field(d, RC_RX_BUF, 5);
    if (dst == rc.our_id && flags == 0x00) {
      rc_build_ack(d, rc.net, rc.our_id, src, ack);
      rc_hal_gdo_mask();
      uint8_t err = cc1101_transmit(ack, RC_ACK_LEN);
      rc_hal_gdo_unmask();
      ack_sent = (err == 0);
    }
  }

  uint16_t local_have = have > RC_RX_BUF_LEN ? RC_RX_BUF_LEN : have;
  memcpy(RC_SCRATCH, RC_RX_BUF, local_have);

  /* Release rx_buf, then drain what the RX_DONE gate held back (in RX_IDLE,
   * so the next GDO2 discards it by resetting rx_have). */
  rc_hal_lock();
  rc.rx_have = 0;
  rc.rx_state = RC_RX_IDLE;
  rx_drain_fifo();
  rc_hal_unlock();

  if (ack_sent) {
    put_p(RC_PSTR("ACK "));
    put_u32(rc_hal_micros());
    rc_hal_putc(' ');
    put_hex(ack, RC_ACK_LEN);
    rc_print_eol();
  }

  uint8_t rssi_raw = 0, lqi_raw = 0;
  if (local_have >= total + 2) {
    rssi_raw = RC_SCRATCH[total];
    lqi_raw = RC_SCRATCH[total + 1];
  }
  uint16_t frame_len = total > 255 ? 255 : total;
  if (frame_len > local_have) frame_len = local_have; /* truncated read: report what we have */

  put_p(RC_PSTR("RX "));
  put_u32(t);
  rc_hal_putc(' ');
  put_dbm10(rc_rssi_dbm10(rssi_raw));
  rc_hal_putc(' ');
  put_u32(lqi_raw & 0x7F);
  rc_hal_putc(' ');
  rc_hal_putc(crc_ok ? '1' : '0');
  rc_hal_putc(' ');
  put_hex(RC_SCRATCH, frame_len);
  if (rc.raw_mode) {
    rc_hal_putc(' ');
    put_hex8(rssi_raw);
    put_hex8(lqi_raw);
  }
  rc_print_eol();
}

void rc_service(void) {
  if (rc.packet_ready) handle_packet();

  /* termoweb_rx 3.5's liveness guard: rx_state stuck at RX_DONE with no
   * packet pending recovers (and says so) after 250 ms. */
  rc_hal_lock();
  uint8_t stuck = rc.rx_state == RC_RX_DONE && !rc.packet_ready;
  uint32_t since = rc.rx_done_micros;
  rc_hal_unlock();
  if (stuck && (rc_hal_micros() - since) > RC_RXRESET_TIMEOUT_US) {
    cc1101_enter_rx();
    rc_hal_lock();
    rc.rx_have = 0;
    rc.rx_state = RC_RX_IDLE;
    rc_hal_unlock();
    rc.rxreset_count++;
    put_p(RC_PSTR("# rxreset n="));
    put_u32(rc.rxreset_count);
    rc_print_eol();
  }
}

/* Re-arm RX after a command changed the chip (T, Y, F): the receiver and
 * the RX state machine restart from RX_IDLE before edges are unmasked, so
 * bytes of a frame cut short are flushed first. */
static void rx_rearm_unmask(void) {
  cc1101_enter_rx();
  rc.rx_have = 0;
  rc.rx_state = RC_RX_IDLE;
  rc.packet_ready = 0;
  rc_hal_gdo_unmask();
}

/* ------------------------------------------------------------ commands */

/* 'T' after its hex line ended (n = 0: empty or bad hex). The bytes go on
 * air exactly as given; framing for a dialect is the host codec's job. */
static void finish_t(uint8_t n) {
  if (n == 0) {
    put_line_p(RC_PSTR("TXERR empty or bad hex"));
    return;
  }
  if (rc.survey_active) {
    put_line_p(RC_PSTR("TXERR survey active"));
    return;
  }
  if (!rc.radio_ok) {
    put_line_p(RC_PSTR("TXERR no radio"));
    return;
  }
  rc_hal_gdo_mask();
  uint8_t err = cc1101_transmit(rc.tx_buf, n);
  rx_rearm_unmask();
  if (err == CC1101_TX_UNDERFLOW) {
    put_line_p(RC_PSTR("TXERR underflow"));
  } else if (err) {
    put_p(RC_PSTR("TXERR marcstate="));
    put_hex8(err);
    rc_print_eol();
  } else {
    put_p(RC_PSTR("TX "));
    put_u32(rc_hal_micros());
    rc_hal_putc(' ');
    put_u32(n);
    rc_hal_putc(' ');
    put_hex(rc.tx_buf, n);
    rc_print_eol();
  }
}

static void finish_i(uint8_t n) {
  if (n == 1) {
    rc.our_id = rc.id_buf[0];
    put_p(RC_PSTR("# id="));
    put_hex8(rc.our_id);
    rc_print_eol();
  } else {
    put_line_p(RC_PSTR("# I? expected 2 hex digits"));
  }
}

/* 'N<hex4>': the network id auto-acks carry. Received frames are never
 * filtered on it, so the host still sees every network's traffic. */
static void finish_n(uint8_t n) {
  if (n == 2) {
    rc.net = (uint16_t)((rc.id_buf[0] << 8) | rc.id_buf[1]);
    put_p(RC_PSTR("# net="));
    put_hex8(rc.id_buf[0]);
    put_hex8(rc.id_buf[1]);
    rc_print_eol();
  } else {
    put_line_p(RC_PSTR("# N? expected 4 hex digits"));
  }
}

/* 'Y0'/'Y1': dialect A/B changes the sync word, length rule, CRC verdict
 * and ack format at once; the sync word is rewritten live. */
static void set_dialect(uint8_t d) {
  rc.dialect = d;
  if (rc.radio_ok) {
    rc_hal_gdo_mask();
    rc_hal_strobe(CC1101_SIDLE);
    cc1101_write_sync();
    rx_rearm_unmask();
  }
  put_p(d == RC_DIALECT_B ? RC_PSTR("# dialect=B sync=") : RC_PSTR("# dialect=A sync="));
  put_sync();
  rc_print_eol();
}

/* 'F<kHz>': retune from IDLE (FREQ must not change while the synthesiser
 * runs); the SRX in cc1101_enter_rx recalibrates. */
static void finish_f(uint8_t ok) {
  if (!ok || rc.f_digits == 0 || rc.f_value < RC_FREQ_KHZ_MIN || rc.f_value > RC_FREQ_KHZ_MAX) {
    put_line_p(RC_PSTR("# F? expected kHz"));
    return;
  }
  rc.freq_khz = rc.f_value;
  if (rc.radio_ok) {
    rc_hal_gdo_mask();
    rc_hal_strobe(CC1101_SIDLE);
    cc1101_write_freq(rc.f_value);
    rx_rearm_unmask();
  }
  uint32_t word = rc_freq_word(rc.f_value);
  put_p(RC_PSTR("# freq="));
  put_freq(rc.freq_khz);
  put_p(RC_PSTR(" word="));
  put_hex8((uint8_t)(word >> 16));
  put_hex8((uint8_t)(word >> 8));
  put_hex8((uint8_t)word);
  rc_print_eol();
}

/* 'R<seconds>': raw survey for up to 600 s; 'R0' ends one early. */
static void finish_r(uint8_t ok) {
  if (!ok || rc.f_digits == 0) {
    put_line_p(RC_PSTR("# R? expected seconds"));
    return;
  }
  if (rc.f_value == 0) {
    if (rc.survey_active) rc_hal_survey(0);
    return;
  }
  if (!rc.radio_ok) {
    rc_print_no_radio();
    return;
  }
  rc_hal_survey((uint16_t)(rc.f_value > RC_SURVEY_MAX_SECS ? RC_SURVEY_MAX_SECS : rc.f_value));
}

void rc_survey_finish(uint16_t bursts) {
  cc1101_init(); /* packet-mode registers, the dialect's sync, the frequency */
  rc_hal_lock();
  rc.rx_have = 0;
  rc.rx_state = RC_RX_IDLE;
  rc.packet_ready = 0;
  rc_hal_unlock();
  cc1101_enter_rx();
  rc_hal_gdo_unmask();
  rc_print_survey_off(bursts);
}

/* A pending multi-byte command failed (timeout, bad byte, overflow). */
static void parse_fail(void) {
  uint8_t s = rc.pstate;
  rc.pstate = RC_P_IDLE;
  switch (s) {
    case RC_P_A:
      put_line_p(RC_PSTR("# A? expected 0 or 1"));
      break;
    case RC_P_Y:
      put_line_p(RC_PSTR("# Y? expected 0 or 1"));
      break;
    case RC_P_HEX_I:
      finish_i(0);
      break;
    case RC_P_HEX_N:
      finish_n(0);
      break;
    case RC_P_HEX_T:
      finish_t(0);
      break;
    case RC_P_DEC_F:
      finish_f(0);
      break;
    case RC_P_DEC_R:
      finish_r(0);
      break;
    default:
      break;
  }
}

static int8_t hex_nibble(char c) {
  if (c >= '0' && c <= '9') return (int8_t)(c - '0');
  if (c >= 'A' && c <= 'F') return (int8_t)(c - 'A' + 10);
  if (c >= 'a' && c <= 'f') return (int8_t)(c - 'a' + 10);
  return -1;
}

/* Decimal argument of 'F' (at most 7 digits) or 'R' (at most 4). */
static uint8_t feed_decimal(char c, uint8_t max_digits) {
  uint8_t is_f = rc.pstate == RC_P_DEC_F;
  if (c == '\n' || c == '\r') {
    rc.pstate = RC_P_IDLE;
    if (is_f)
      finish_f(1);
    else
      finish_r(1);
    return 1;
  }
  if (c >= '0' && c <= '9' && rc.f_digits < max_digits) {
    rc.f_value = rc.f_value * 10 + (uint32_t)(c - '0');
    rc.f_digits++;
    return 0;
  }
  parse_fail();
  return 0;
}

/* read_hex_line(): newline ends the line (success only on an even nibble
 * count); any other non-hex byte, or a byte past the maximum, fails it
 * with that byte consumed. */
static uint8_t feed_hex(char c) {
  uint8_t is_t = rc.pstate == RC_P_HEX_T;
  uint8_t *out = is_t ? rc.tx_buf : rc.id_buf;
  uint8_t max = is_t ? RC_TX_MAX_LEN : (rc.pstate == RC_P_HEX_N ? 2 : 1);
  if (c == '\n' || c == '\r') {
    uint8_t n = rc.hi < 0 ? rc.hex_n : 0;
    uint8_t done = rc.pstate;
    rc.pstate = RC_P_IDLE;
    if (is_t) {
      finish_t(n);
      return 1;
    }
    if (done == RC_P_HEX_N)
      finish_n(n);
    else
      finish_i(n);
    return 0;
  }
  int8_t v = hex_nibble(c);
  if (v < 0) {
    parse_fail();
    return is_t;
  }
  if (rc.hi < 0) {
    rc.hi = v;
  } else {
    if (rc.hex_n >= max) {
      parse_fail();
      return is_t;
    }
    out[rc.hex_n++] = (uint8_t)((rc.hi << 4) | v);
    rc.hi = -1;
  }
  return 0;
}

uint8_t rc_feed(char c) {
  switch (rc.pstate) {
    case RC_P_IDLE:
      break;
    case RC_P_A:
      rc.pstate = RC_P_IDLE;
      if (c == '0' || c == '1') {
        rc.auto_ack = (c == '1');
        put_line_p(rc.auto_ack ? RC_PSTR("# autoack=on") : RC_PSTR("# autoack=off"));
      } else {
        put_line_p(RC_PSTR("# A? expected 0 or 1"));
      }
      return 0;
    case RC_P_Y:
      rc.pstate = RC_P_IDLE;
      if (c == '0' || c == '1') {
        set_dialect(c == '1' ? RC_DIALECT_B : RC_DIALECT_A);
        return 1;
      }
      put_line_p(RC_PSTR("# Y? expected 0 or 1"));
      return 0;
    case RC_P_HEX_I:
    case RC_P_HEX_N:
    case RC_P_HEX_T:
      return feed_hex(c);
    case RC_P_DEC_F:
      return feed_decimal(c, 7);
    case RC_P_DEC_R:
      return feed_decimal(c, 4);
    default:
      rc.pstate = RC_P_IDLE;
      return 0;
  }

  /* Idle: command dispatch. Unknown bytes (including the '\n' that ends
   * every command line) are ignored. */
  rc.cmd_start_us = rc_hal_micros();
  switch (c) {
    case 'V':
      rc_print_banner();
      break;
    case 'Q':
      rc_print_query();
      break;
    case 'X':
      rc.raw_mode = !rc.raw_mode;
      put_line_p(rc.raw_mode ? RC_PSTR("# raw=on") : RC_PSTR("# raw=off"));
      break;
    case 'A':
      rc.pstate = RC_P_A;
      break;
    case 'Y':
      rc.pstate = RC_P_Y;
      break;
    case 'I':
    case 'N':
    case 'T':
      rc.pstate = c == 'T' ? RC_P_HEX_T : (c == 'N' ? RC_P_HEX_N : RC_P_HEX_I);
      rc.hi = -1;
      rc.hex_n = 0;
      break;
    case 'F':
    case 'R':
      rc.pstate = c == 'F' ? RC_P_DEC_F : RC_P_DEC_R;
      rc.f_value = 0;
      rc.f_digits = 0;
      break;
    case 'D': {
      uint8_t marcstate = rc_hal_read_status(CC1101_MARCSTATE);
      uint8_t pktstatus = rc_hal_read_status(CC1101_PKTSTATUS);
      uint8_t rxbytes = rc_hal_read_status(CC1101_RXBYTES);
      rc_hal_lock();
      uint16_t syncs = rc.sync_count;
      rc_hal_unlock();
      put_p(RC_PSTR("# marcstate="));
      put_hex8(marcstate);
      put_p(RC_PSTR(" pktstatus="));
      put_hex8(pktstatus);
      put_p(RC_PSTR(" rxbytes="));
      put_hex8(rxbytes);
      put_p(RC_PSTR(" syncs="));
      put_u32(syncs);
      put_p(RC_PSTR(" rxreset="));
      put_u32(rc.rxreset_count);
      rc_print_eol();
      break;
    }
    default:
      break;
  }
  return 0;
}

/* Timeouts are judged only when no byte is waiting, exactly like the old
 * blocking reads, which looked at the clock only when the UART was empty. */
void rc_poll(void) {
  if (rc.pstate != RC_P_IDLE && (rc_hal_micros() - rc.cmd_start_us) > RC_CMD_TIMEOUT_US) parse_fail();
}

void rc_begin(void) {
  rc_hal_gdo_mask();
  rc.pstate = RC_P_IDLE;
  rc.raw_mode = 0;
  rc.auto_ack = 0;
  rc.dialect = RC_DIALECT_A;
  rc.net = RC_DEFAULT_NET;
  rc.our_id = RC_DEFAULT_ID;
  rc.sync_count = 0;
  rc.rxreset_count = 0;
  rc.rx_have = 0;
  rc.rx_total = 0;
  rc.rx_state = RC_RX_IDLE;
  rc.packet_ready = 0;
  rc.freq_khz = rc.default_khz ? rc.default_khz : RC_DEFAULT_FREQ_KHZ;
  rc.radio_ok = cc1101_present();
  if (!rc.radio_ok) {
    /* Nothing to configure or receive from: edges stay masked, and T, R and
     * the receive path stay off until the next reset probes again. */
    rc_print_banner();
    rc_print_no_radio();
    return;
  }
  cc1101_init();
  rc_print_banner();
  cc1101_enter_rx();
  rc_hal_gdo_unmask();
}

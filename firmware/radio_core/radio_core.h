/* radio_core.h: the portable part of the termoweb_rx radio firmware.
 *
 * One C implementation of everything the nanoCUL and the ESP32 radio share:
 *   - the two on-air dialects: where a frame ends (byte 0), its CRC verdict,
 *     the PN9 keystream, and the 8-byte link ack built inside a heater's
 *     few-millisecond ack window;
 *   - the line protocol: the command parser (an incremental state machine
 *     fed one byte at a time) and every reply line, byte for byte;
 *   - the receive state machine (RX FIFO drain, dynamic length, packet
 *     handler with auto-ack, liveness guard) on top of cc1101.c.
 * Hardware access goes through rc_hal.h. Plain C99, no stdio, no heap,
 * no floating point; reply strings live in flash on AVR (RC_PSTR).
 *
 * Dialects (the host picks one at runtime with 'Y'):
 *   A (default after every reset): sync 2DE5; byte 0 = total - 3; every
 *     byte PN9-scrambled; CRC-16/CCITT (poly 0x1021, init 0x1D0F, final
 *     XOR 0xFFFF) over the descrambled bytes [0, total-3]; network id set
 *     with 'N'.
 *   B: sync 2DD4; byte 0 = total; bytes in the clear; CRC-16/MODBUS
 *     (reflected 0xA001, init 0xFFFF) over [0, total-3]; network id set
 *     with 'N'.
 *   In both, the CRC is sent high byte first, the bytes after byte 0 are
 *   net(2) src dst flags path(5) tag payload crc(2), and an ack is
 *   net, acker, acked sender, flags 0x80, CRC.
 *
 * Line protocol (lines end "\r\n"): see firmware/README.md for the table.
 */
#ifndef RADIO_CORE_H
#define RADIO_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version reported in the banner and Q: "<protocol>-<port>". The protocol
 * part (3.7) is the line protocol and feature level both ports share; the
 * suffix names the hardware port. Set by each port's build. */
#ifndef RC_VERSION
#error "RC_VERSION is not defined: build with the port's Makefile"
#endif

/* PATABLE value. 0xC0 = +10 dBm at 868 MHz, the setting heaters were proven
 * to ack in the first live test; 0x50 (0 dBm) was the old Makefile default. */
#ifndef RC_PA
#define RC_PA 0xC0
#endif

/* Strings in flash on AVR (2 KB RAM would not hold them); plain elsewhere. */
#ifdef __AVR__
#include <avr/pgmspace.h>
#define RC_PROGMEM PROGMEM
#define RC_PSTR(s) PSTR(s)
#define RC_RD(p) pgm_read_byte(p)
#else
#define RC_PROGMEM
#define RC_PSTR(s) (s)
#define RC_RD(p) (*(const uint8_t *)(p))
#endif

#define RC_DIALECT_A 0
#define RC_DIALECT_B 1
#define RC_SYNC1 0x2D
#define RC_SYNC0_A 0xE5
#define RC_SYNC0_B 0xD4
#define RC_DEFAULT_NET 0x0000 /* none until the host sends 'N' */
#define RC_DEFAULT_ID 0x01
#define RC_DEFAULT_FREQ_KHZ 869525UL
#define RC_FREQ_KHZ_MIN 779000UL /* the CC1101's 868/915 MHz synthesiser band */
#define RC_FREQ_KHZ_MAX 928000UL
#define RC_PKT_LEN 64 /* PKTLEN between receptions (a don't-care in infinite mode) */
#define RC_TX_MAX_LEN 255
#define RC_RX_TOTAL_MIN 8 /* frame totals outside [8, 255] are noise */
#define RC_RX_TOTAL_MAX 255
#define RC_RX_BUF_LEN (RC_RX_TOTAL_MAX + 2) /* + the 2 APPEND_STATUS bytes */
#define RC_ACK_LEN 8
#define RC_CMD_TIMEOUT_US 1000000UL
#define RC_RXRESET_TIMEOUT_US 250000UL
#define RC_SURVEY_MAX_SECS 600
#define RC_SURVEY_MARGIN_DBM10 80 /* trigger 8 dB above the noise floor */

enum { RC_RX_IDLE = 0, RC_RX_LENPENDING, RC_RX_FIXED, RC_RX_DONE };
enum { RC_P_IDLE = 0, RC_P_A, RC_P_Y, RC_P_HEX_I, RC_P_HEX_N, RC_P_HEX_T, RC_P_DEC_F, RC_P_DEC_R };

struct rc_state {
  /* rx_buf (first half) and the packet handler's snapshot (second half).
   * A raw survey turns packet RX off, so its run buffer overlays both. */
  union {
    uint8_t b[2 * RC_RX_BUF_LEN];
    uint16_t runs[RC_RX_BUF_LEN];
  } mem;
  uint8_t tx_buf[RC_TX_MAX_LEN]; /* 'T' payload, filled while the line arrives */

  /* Receive state, shared with the GDO handlers. */
  volatile uint16_t rx_have;
  volatile uint16_t rx_total;
  volatile uint8_t rx_state;
  volatile uint8_t packet_ready;
  volatile uint32_t rx_done_micros;
  volatile uint16_t sync_count; /* 'D' syncs=: GDO0 packet-done edges, noise included */
  uint16_t rxreset_count;
  uint32_t frames_ok; /* CRC-valid frames since boot */

  /* Session state; back to the defaults on every reset (rc_begin). */
  uint8_t auto_ack;
  uint8_t dialect;
  uint8_t our_id;
  uint8_t raw_mode;
  uint8_t radio_ok; /* a CC1101 answered the probe at the last reset */
  volatile uint8_t survey_active; /* set and cleared by the platform's survey */
  uint16_t net;
  uint32_t freq_khz;
  uint32_t default_khz; /* 0: RC_DEFAULT_FREQ_KHZ */
  const char *mac;      /* RAM string for mac=, or NULL to leave the field out */

  /* Command parser. */
  uint8_t pstate;
  int8_t hi;
  uint8_t hex_n;
  uint8_t f_digits;
  uint32_t f_value;
  uint32_t cmd_start_us;
  uint8_t id_buf[2];
};

extern struct rc_state rc;

#define RC_RX_BUF (rc.mem.b)
#define RC_SCRATCH (rc.mem.b + RC_RX_BUF_LEN)

/* ---- on-air protocol (pure functions) ---- */
uint16_t rc_crc_ccitt(const uint8_t *data, uint16_t len);
uint16_t rc_crc_modbus(const uint8_t *data, uint16_t len);
/* PN9 keystream, generated on the fly (seed 0xFF, MSB first). */
struct rc_pn9 {
  uint8_t reg;
  uint8_t fb;
};
void rc_pn9_init(struct rc_pn9 *p);
uint8_t rc_pn9_next(struct rc_pn9 *p); /* current keystream byte, then advance */
/* Total on-air length from byte 0; 0 when outside [8, 255]. */
uint16_t rc_frame_total(uint8_t dialect, uint8_t byte0);
/* Does this on-air frame carry a valid length byte and CRC in `dialect`? */
uint8_t rc_frame_crc_ok(uint8_t dialect, const uint8_t *air, uint16_t total);
/* Logical byte i (3 = src, 4 = dst, 5 = flags) of an on-air frame. */
uint8_t rc_field(uint8_t dialect, const uint8_t *air, uint8_t i);
void rc_build_ack(uint8_t dialect, uint16_t net, uint8_t acker, uint8_t src, uint8_t out[RC_ACK_LEN]);
/* FREQ word = round(kHz * 2^16 / 26 MHz); 869525 -> 0x21717A. */
uint32_t rc_freq_word(uint32_t khz);
/* RSSI status byte -> dBm x10 (raw/2 - 74). */
int16_t rc_rssi_dbm10(uint8_t raw);
/* Ack builders and CRC checkers against fixed vectors; 1 = pass. */
uint8_t rc_self_test(void);

/* ---- session ---- */
/* Power-on / new-connection reset: session defaults, chip probe, chip
 * init, banner, receiver armed. GDO edges are masked throughout. Without a
 * CC1101 (cc1101_present() fails) it prints the banner with radio=missing
 * and "# error: no CC1101 radio found", and leaves the edges masked. */
void rc_begin(void);
/* One input byte. Returns 1 when it completed a command that used the
 * radio, so a port can service GDO events before parsing more. */
uint8_t rc_feed(char c);
/* Call when no input byte is waiting: ends a half-received command after
 * 1 s, exactly like the old blocking reads. */
void rc_poll(void);
/* Main-loop work: the packet handler (auto-ack, RX/ACK lines) and the
 * stuck-receiver guard. */
void rc_service(void);
/* GDO handlers: GDO2 rose (RX FIFO at threshold), GDO0 fell (packet end,
 * edge_us = when). Interrupt-safe on AVR. */
void rc_on_gdo2(void);
void rc_on_gdo0(uint32_t edge_us);

/* ---- lines ---- */
void rc_print_banner(void);
void rc_print_query(void);
void rc_print_overrun(void);
void rc_print_no_radio(void);
void rc_print_self_test_failed(void);
/* Raw survey lines. dBm values are x10 (one decimal place is printed). */
void rc_print_survey_on(uint16_t secs, int16_t floor_dbm10, int16_t thr_dbm10);
void rc_print_survey_off(uint16_t bursts);
void rc_print_rawb(uint16_t burst, int16_t peak_dbm10);
void rc_print_raw_start(uint16_t burst);
void rc_print_raw_run(uint8_t level, uint16_t us);
void rc_print_rawe(uint16_t burst, uint16_t runs);
void rc_print_eol(void);
/* Back to packet mode after a survey: chip re-init, receiver armed, edges
 * unmasked, "# survey off bursts=<n>". The platform has cleared
 * rc.survey_active and stopped its edge capture first. */
void rc_survey_finish(uint16_t bursts);

#ifdef __cplusplus
}
#endif

#endif

/* termoweb_radio.h: a transparent CC1101 radio for Sun Ray / Termoweb RF
 * heaters on an ESP32 or ESP32-S3, running inside ESPHome and served over
 * raw TCP (port 2323). It is the ESP32 port of termoweb_rx 3.5, the nanoCUL868
 * firmware in ha-termoweb-local (firmware/termoweb_rx), and keeps its line
 * protocol, so a host that opens socket://<device>:2323 with pyserial's
 * serial_for_url sees the same lines a nanoCUL prints over USB.
 *
 * The firmware is TRANSPARENT: RX lines carry the on-air bytes exactly as
 * received and 'T' sends the given bytes verbatim. Everything that knows
 * what a frame means lives in the host's codec. The firmware only needs to
 * know the on-air "dialect" for the three things that cannot wait for a host
 * round trip: where a frame ends (its length byte), whether its CRC holds,
 * and how to build the 8-byte link ack inside a heater's few-millisecond
 * ack window. The host picks the dialect at runtime with 'Y':
 *
 *   dialect A (termoweb_rx's; default after every reset, as on a nanoCUL):
 *     sync 2DE5; byte 0 = (total - 3); every byte PN9-scrambled; CRC-16/CCITT
 *     (poly 0x1021, init 0x1D0F, final XOR 0xFFFF) over the descrambled bytes
 *     [0, total-3], high byte first; network id 1B 30.
 *   dialect B (captured from a second network: 68 frames, all CRC-valid):
 *     sync 2DD4; byte 0 = total (itself and the CRC included); bytes in the
 *     clear; CRC-16/MODBUS (reflected 0xA001, init 0xFFFF, no final XOR) over
 *     bytes [0, total-3], high byte first; its network id is per installation (the host sets it with 'N').
 *   In both, the bytes after byte 0 are net(2) src dst flags path(5) tag
 *   payload crc(2), and an ack is net, acker, acked sender, flags 0x80, CRC.
 *
 * Line protocol (lines end "\r\n", as termoweb_rx's uart_putc_stream sends):
 *   connect / 'V' -> "# termoweb_rx 3.6-esp32 freq=869.525 rate=9.6k sync=2DE5 mode=dynamic tx=paC0 mac=<mac>"
 *   'Q'        -> "# Q termoweb_rx 3.6-esp32 freq=... pa=... sync=... mode=dynamic autoack=<on/off> id=<hex>
 *                  dialect=<A|B> net=<hex4> mac=<AA:BB:CC:DD:EE:FF>" (one line)
 *   'X'        -> toggle raw mode (appends the 2 raw status bytes as hex)
 *   'A0'/'A1'  -> auto-ack off/on (default off) -> "# autoack=on|off"
 *   'I<hex>'   -> our station id (default 01), used by auto-ack -> "# id=<hex>"
 *   'T<hex>'   -> transmit these bytes verbatim, up to 255; "TX <micros> <n> <hex>"
 *                 or "TXERR empty or bad hex" / "TXERR underflow" / "TXERR marcstate=<hex>"
 *   'D'        -> "# marcstate=.. pktstatus=.. rxbytes=.. syncs=<n> rxreset=<n>"
 *   received frame -> "RX <micros> <rssi_dbm> <lqi> <crc_ok> <hex bytes as received>"
 *                     crc_ok is the current dialect's CRC verdict (1 = valid)
 *   sent ack        -> "ACK <micros> <hex bytes as sent>"
 * ESP32-only commands (none of them changes a termoweb_rx reply):
 *   'Y0'/'Y1'  -> dialect A/B; retunes the sync word live.
 *                 "# dialect=A sync=2DE5" / "# dialect=B sync=2DD4", or "# Y? expected 0 or 1"
 *   'N<hex4>'  -> network id auto-acks carry (default 1B30) -> "# net=1234",
 *                 or "# N? expected 4 hex digits"
 *   'F<kHz>'   -> retune the carrier, e.g. "F869525" -> "# freq=869.525 word=21717A",
 *                 or "# F? expected kHz" (accepted range 779000-928000)
 * Dialect, network id, frequency, auto-ack, station id and raw mode all go
 * back to their defaults on every new connection (see "Connection
 * semantics"), so a host sends Y/N/I/A1 after each connect.
 *
 * mac= is the ESP32's WiFi station MAC: a stable id for the host to tell
 * radios apart, which survives IP address changes.
 *
 * Architecture, and why it differs from termoweb_rx's shape:
 *
 * termoweb_rx is one AVR main loop plus three ISRs, and its INT0/INT1 ISRs
 * do SPI transactions themselves. On the ESP32, spi_master transactions are
 * not callable from an ISR, and the GPIO ISRs must stay short, so the AVR
 * ISR bodies move into one FreeRTOS "radio" task that owns the CC1101
 * exclusively: the GPIO ISRs only timestamp the edge and set a task
 * notification bit, and the radio task then runs exactly the code the AVR
 * ISR would have run (on_gdo2() / on_gdo0() below). Because the packet
 * handler (main.c's packet_ready block) also runs in that same task, the
 * AVR's cli()/sei() pairs around the rx_state/packet_ready pair have no
 * counterpart here: nothing else ever touches that state, so there is no
 * interleaving to guard against.
 *
 * The radio task is pinned to whichever core ESPHome's own loop task is NOT
 * running on (found at start() time, since start() is called from that loop
 * task), so a slow ESPHome component can never delay an auto-ack.
 *
 * main.c's blocking command reads (uart_getc_blocking for 'A', read_hex_line
 * for 'I'/'T') would stall the radio task for up to 1 s on a slow client, so
 * the command parser is an incremental state machine fed one byte at a time
 * with the SAME semantics: the same 1 s timeout measured from the command
 * letter, the same "any non-hex byte or an odd nibble count fails the whole
 * line" rule, and on failure the offending byte is consumed and anything
 * after it is parsed as fresh commands, exactly as main.c's main loop would
 * see the rest of a bad line. The one observable difference is an
 * improvement: a frame that completes while a 'T' line is still arriving is
 * printed rather than silently dropped by the 'T' handler's packet_ready = 0.
 *
 * Two more tasks handle TCP: a listener (accept + recv, feeding the command
 * ring) and a writer (drains the output ring into send()). The radio task
 * never blocks on the network: it appends whole lines to the output ring
 * and moves on, dropping a whole line (never half of one, which would hand
 * the host a truncated RX hex string) if a stalled client has let the ring
 * fill up.
 *
 * Connection semantics: ONE client at a time. A new connection replaces the
 * old one (the old socket is closed), matching a serial port, which only
 * one process can hold. Each new connection gets the same treatment a DTR
 * reset gives the nanoCUL on port open: the CC1101 is re-initialised, all
 * session state (auto-ack, station id, raw mode, dialect, network id,
 * frequency, counters, half-parsed command) goes back to its power-on
 * default, and the boot banner is the first line the new client receives.
 * Only the micros clock keeps running (the AVR's restarts at 0 on reset);
 * hosts only ever subtract two micros values, so this is invisible to them.
 *
 * Use from ESPHome YAML: optionally termoweb::radio.set_pins(...), then
 * termoweb::radio.start() from on_boot (priority -100). No `spi:` or
 * `cc1101:` component may share the bus: this file owns SPI2 and the chip.
 */
#pragma once

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_mac.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"

#include "esphome/core/log.h"

namespace termoweb {

static const char *const TAG = "termoweb_radio";

/* Reported in the banner and Q. The host only checks that "termoweb_rx"
 * appears in either line; the "-esp32" suffix tells a human (and a host
 * that cares) that the Y/N/F extensions and mac= are available. */
#ifndef TERMOWEB_RADIO_VERSION
#define TERMOWEB_RADIO_VERSION "3.6-esp32"
#endif

/* PATABLE value: 0xC0 = +10 dBm on 868 MHz (CC1101 datasheet). The nanoCUL
 * Makefile defaults to 0x50 (0 dBm) but the deployed stick is built with
 * PA=0xC0, the setting the first live test proved heaters ack (firmware
 * README "Transmit (3.1)"), so that is this port's default. */
#ifndef TERMOWEB_RADIO_PA
#define TERMOWEB_RADIO_PA 0xC0
#endif

/* Default wiring: ESP32-S3 (N16R8) to the CC1101 module. Overridable at
 * runtime with set_pins() before start(), which is how the YAML's
 * substitutions reach this file. */
static constexpr int DEFAULT_PIN_SCK = 12;
static constexpr int DEFAULT_PIN_MOSI = 11;
static constexpr int DEFAULT_PIN_MISO = 14;
static constexpr int DEFAULT_PIN_CSN = 10;
static constexpr int DEFAULT_PIN_GDO0 = 9;  /* main.c INT1: packet done (falling edge) */
static constexpr int DEFAULT_PIN_GDO2 = 13; /* main.c INT0: RX FIFO at threshold (rising edge) */

static constexpr uint16_t TCP_PORT = 2323;

/* Dialects (see the file header). Sync word high byte is 0x2D in both. */
enum Dialect : uint8_t { DIALECT_A = 0, DIALECT_B = 1 };
static constexpr uint8_t SYNC1_VAL = 0x2D;
static constexpr uint8_t SYNC0_A = 0xE5;
static constexpr uint8_t SYNC0_B = 0xD4;
/* Network id auto-acks carry after a reset: termoweb_rx's own, 1B 30. */
static constexpr uint16_t DEFAULT_NET = 0x1B30;

static constexpr uint8_t PKT_LEN = 64;
static constexpr uint8_t PA_VAL = TERMOWEB_RADIO_PA;
static constexpr uint16_t TX_MAX_LEN = 255;

/* Carrier. 869.525 MHz is the gateway label value; FREQ = round(f * 2^16 /
 * 26 MHz) = 0x21717A (869.524963 MHz actual), cc1101.c's FREQ2/1/0. The
 * 779-928 MHz window accepted by 'F' is the CC1101's own 868/915 MHz
 * synthesiser band; the register table (PA value, FSCAL, TEST regs) is the
 * 868 MHz one, so the lower 315/433 MHz bands are deliberately refused. */
static constexpr uint32_t DEFAULT_FREQ_KHZ = 869525;
static constexpr uint32_t FREQ_KHZ_MIN = 779000;
static constexpr uint32_t FREQ_KHZ_MAX = 928000;

/* Config register addresses (CC1101 datasheet Table 45), cc1101.h. */
static constexpr uint8_t CC1101_IOCFG2 = 0x00;
static constexpr uint8_t CC1101_IOCFG0 = 0x02;
static constexpr uint8_t CC1101_FIFOTHR = 0x03;
static constexpr uint8_t CC1101_SYNC1 = 0x04;
static constexpr uint8_t CC1101_SYNC0 = 0x05;
static constexpr uint8_t CC1101_PKTLEN = 0x06;
static constexpr uint8_t CC1101_PKTCTRL1 = 0x07;
static constexpr uint8_t CC1101_PKTCTRL0 = 0x08;
static constexpr uint8_t CC1101_ADDR = 0x09;
static constexpr uint8_t CC1101_CHANNR = 0x0A;
static constexpr uint8_t CC1101_FSCTRL1 = 0x0B;
static constexpr uint8_t CC1101_FSCTRL0 = 0x0C;
static constexpr uint8_t CC1101_FREQ2 = 0x0D;
static constexpr uint8_t CC1101_FREQ1 = 0x0E;
static constexpr uint8_t CC1101_FREQ0 = 0x0F;
static constexpr uint8_t CC1101_MDMCFG4 = 0x10;
static constexpr uint8_t CC1101_MDMCFG3 = 0x11;
static constexpr uint8_t CC1101_MDMCFG2 = 0x12;
static constexpr uint8_t CC1101_MDMCFG1 = 0x13;
static constexpr uint8_t CC1101_MDMCFG0 = 0x14;
static constexpr uint8_t CC1101_DEVIATN = 0x15;
static constexpr uint8_t CC1101_MCSM1 = 0x17;
static constexpr uint8_t CC1101_MCSM0 = 0x18;
static constexpr uint8_t CC1101_FOCCFG = 0x19;
static constexpr uint8_t CC1101_BSCFG = 0x1A;
static constexpr uint8_t CC1101_AGCCTRL2 = 0x1B;
static constexpr uint8_t CC1101_AGCCTRL1 = 0x1C;
static constexpr uint8_t CC1101_AGCCTRL0 = 0x1D;
static constexpr uint8_t CC1101_FREND1 = 0x21;
static constexpr uint8_t CC1101_FREND0 = 0x22;
static constexpr uint8_t CC1101_FSCAL3 = 0x23;
static constexpr uint8_t CC1101_FSCAL2 = 0x24;
static constexpr uint8_t CC1101_FSCAL1 = 0x25;
static constexpr uint8_t CC1101_FSCAL0 = 0x26;
static constexpr uint8_t CC1101_FSTEST = 0x29;
static constexpr uint8_t CC1101_TEST2 = 0x2C;
static constexpr uint8_t CC1101_TEST1 = 0x2D;
static constexpr uint8_t CC1101_TEST0 = 0x2E;

/* Status registers (read one at a time with the burst bit set; see cc1101.h
 * for why they cannot be batched). PARTNUM/VERSION are only read once, at
 * start, to log the chip identity (expected 0x00/0x14). */
static constexpr uint8_t CC1101_PARTNUM = 0x30;
static constexpr uint8_t CC1101_VERSION = 0x31;
static constexpr uint8_t CC1101_MARCSTATE = 0x35;
static constexpr uint8_t CC1101_PKTSTATUS = 0x38;
static constexpr uint8_t CC1101_TXBYTES = 0x3A;
static constexpr uint8_t CC1101_RXBYTES = 0x3B;

/* Command strobes. */
static constexpr uint8_t CC1101_SRES = 0x30;
static constexpr uint8_t CC1101_SCAL = 0x33;
static constexpr uint8_t CC1101_SRX = 0x34;
static constexpr uint8_t CC1101_STX = 0x35;
static constexpr uint8_t CC1101_SIDLE = 0x36;
static constexpr uint8_t CC1101_SFRX = 0x3A;
static constexpr uint8_t CC1101_SFTX = 0x3B;
static constexpr uint8_t CC1101_PATABLE = 0x3E;

/* SPI header bits. */
static constexpr uint8_t CC1101_WRITE_BURST = 0x40;
static constexpr uint8_t CC1101_READ_BURST = 0xC0;
static constexpr uint8_t CC1101_FIFO_ADDR = 0x3F;

/* cc1101.h: kept outside the 5-bit MARCSTATE range so an underflow can be
 * told apart from "timed out sitting in MARCSTATE 0x16". */
static constexpr uint8_t CC1101_TX_UNDERFLOW = 0xFF;
static constexpr uint8_t MARCSTATE_TXFIFO_UNDERFLOW = 0x16;
/* cc1101.c: TXBYTES reads back 0 when the FIFO is exactly full (silicon
 * erratum), so fills stop one byte short of 64. */
static constexpr uint8_t CC1101_TX_FIFO_SIZE = 64;
static constexpr uint8_t CC1101_TX_INITIAL_FILL = 63;
static constexpr uint8_t CC1101_TX_REFILL_THRESHOLD = 32;

/* main.c: frame totals outside [8, 255] are noise; buffer is the 255-byte
 * ceiling plus the 2 APPEND_STATUS bytes. */
static constexpr uint16_t RX_TOTAL_MIN = 8;
static constexpr uint16_t RX_TOTAL_MAX = 255;
static constexpr uint16_t RX_BUF_LEN = RX_TOTAL_MAX + 2;

/* main.c 3.5 liveness guard threshold; see its use in radio_loop(). */
static constexpr uint32_t RXRESET_TIMEOUT_US = 250000UL;
/* main.c's uart_getc_blocking / read_hex_line timeout. */
static constexpr uint32_t CMD_TIMEOUT_US = 1000000UL;

/* Dialect A's full 255-byte PN9 keystream, _vendored_frame.keystream()
 * transcribed (8-bit register formulation, seed 0xFF, bytes MSB-first).
 * main.c only needed the first 8 bytes (for the ack); checking a received
 * frame's CRC needs one keystream byte per frame byte, up to the 255-byte
 * ceiling. Built at compile time; its first 8 bytes are main.c's
 * FF 87 B8 59 B7 A1 CC 24. */
struct Keystream {
  uint8_t b[255];
  constexpr Keystream() : b{} {
    uint8_t reg = 0xFF, fb = 1;
    for (int i = 0; i < 255; i++) {
      b[i] = reg;
      for (int k = 0; k < 8; k++) {
        uint8_t fb_old = fb;
        fb = (uint8_t) ((((reg ^ ((reg << 5) & 0xFF)) >> 7) & 1));
        reg = (uint8_t) (((reg << 1) | fb_old) & 0xFF);
      }
    }
  }
  constexpr uint8_t operator[](int i) const { return b[i]; }
};
static constexpr Keystream KEYSTREAM{};
static_assert(KEYSTREAM[0] == 0xFF && KEYSTREAM[1] == 0x87 && KEYSTREAM[7] == 0x24 && KEYSTREAM[15] == 0x50,
              "PN9 keystream must match _vendored_frame.keystream()");

/* Register table, cc1101.c's CONFIG[] verbatim (see its comments there and
 * firmware/README.md for every value's derivation), except FREQ2/1/0 and
 * SYNC1/0, which are written separately from runtime state (write_freq_,
 * write_sync_) so 'F' and 'Y' can change them; at the defaults they are the
 * same 0x21 0x71 0x7A and 0x2D 0xE5. */
struct RegVal {
  uint8_t addr;
  uint8_t val;
};
static constexpr RegVal CONFIG[] = {
    {CC1101_IOCFG2, 0x00},   /* GDO2: RX FIFO at/above threshold -> drives on_gdo2 */
    {CC1101_IOCFG0, 0x06},   /* GDO0: asserts on sync, deasserts at end of packet -> on_gdo0 on the falling edge */
    {CC1101_FIFOTHR, 0x40},  /* ADC_RETENTION + 4-byte RX threshold, fastest read of the length byte */
    {CC1101_PKTLEN, PKT_LEN}, /* don't-care in infinite mode; rewritten per frame */
    {CC1101_PKTCTRL1, 0x04}, /* APPEND_STATUS: RSSI + LQI/CRC_OK follow each packet */
    {CC1101_PKTCTRL0, 0x02}, /* infinite length until byte 0 reveals the total; CRC and whitening off */
    {CC1101_ADDR, 0x00},
    {CC1101_CHANNR, 0x00},
    {CC1101_FSCTRL1, 0x06},
    {CC1101_FSCTRL0, 0x00},
    {CC1101_MDMCFG4, 0x68},  /* 270.8 kHz RX BW, DRATE_E=8 */
    {CC1101_MDMCFG3, 0x83},  /* 9596 bps */
    {CC1101_MDMCFG2, 0x02},  /* 2-FSK, 16/16 sync */
    {CC1101_MDMCFG1, 0x42},  /* 8-byte TX preamble */
    {CC1101_MDMCFG0, 0xF8},
    {CC1101_DEVIATN, 0x50},  /* 50.8 kHz deviation */
    {CC1101_MCSM1, 0x30},    /* IDLE after packet; firmware re-arms RX itself */
    {CC1101_MCSM0, 0x18},    /* autocal on IDLE->RX, so every SRX (and 'F') recalibrates */
    {CC1101_FOCCFG, 0x16},
    {CC1101_BSCFG, 0x1C},
    {CC1101_AGCCTRL2, 0xC7},
    {CC1101_AGCCTRL1, 0x00},
    {CC1101_AGCCTRL0, 0xB2},
    {CC1101_FREND1, 0x56},
    {CC1101_FREND0, 0x11},   /* PA index 1: PATABLE[0..1], both written PA_VAL */
    {CC1101_FSCAL3, 0xE9},
    {CC1101_FSCAL2, 0x2A},
    {CC1101_FSCAL1, 0x00},
    {CC1101_FSCAL0, 0x1F},
    {CC1101_FSTEST, 0x59},
    {CC1101_TEST2, 0x81},
    {CC1101_TEST1, 0x35},
    {CC1101_TEST0, 0x09},
};

/* Task notification bits for the radio task. The GPIO ISRs set GDO0/GDO2;
 * the TCP listener sets CMD (bytes queued) and CONNECT (new client);
 * set_frequency_khz() sets RETUNE. */
static constexpr uint32_t EV_GDO0 = 1u << 0;
static constexpr uint32_t EV_GDO2 = 1u << 1;
static constexpr uint32_t EV_CMD = 1u << 2;
static constexpr uint32_t EV_CONNECT = 1u << 3;
static constexpr uint32_t EV_RETUNE = 1u << 4;

/* Mutex-guarded byte ring shared by exactly two tasks. Used for the command
 * path (listener -> radio, main.c's 3.4 UART RX ring) and the output path
 * (radio -> writer, main.c's blocking printf). A FreeRTOS stream buffer was
 * not used because the listener also has to clear both rings on a new
 * connection, which a stream buffer forbids while its reader is blocked. */
template<size_t N> class ByteRing {
  static_assert((N & (N - 1)) == 0, "power of two, so the index wrap is a mask");

 public:
  void init() { this->mtx_ = xSemaphoreCreateMutex(); }
  /* All-or-nothing push: an output line is either queued whole or dropped
   * whole, so the client never receives a truncated RX line. */
  bool push_all(const uint8_t *data, size_t len) {
    xSemaphoreTake(this->mtx_, portMAX_DELAY);
    bool ok = N - (this->head_ - this->tail_) >= len;
    if (ok) this->copy_in_(data, len);
    xSemaphoreGive(this->mtx_);
    return ok;
  }
  /* Partial push for the byte-stream command path; returns bytes taken. */
  size_t push_some(const uint8_t *data, size_t len) {
    xSemaphoreTake(this->mtx_, portMAX_DELAY);
    size_t room = N - (this->head_ - this->tail_);
    if (len > room) len = room;
    this->copy_in_(data, len);
    xSemaphoreGive(this->mtx_);
    return len;
  }
  size_t pop(uint8_t *out, size_t max) {
    xSemaphoreTake(this->mtx_, portMAX_DELAY);
    size_t n = this->head_ - this->tail_;
    if (n > max) n = max;
    for (size_t i = 0; i < n; i++) out[i] = this->buf_[(this->tail_ + i) & (N - 1)];
    this->tail_ += n;
    xSemaphoreGive(this->mtx_);
    return n;
  }
  bool empty() {
    xSemaphoreTake(this->mtx_, portMAX_DELAY);
    bool e = this->head_ == this->tail_;
    xSemaphoreGive(this->mtx_);
    return e;
  }
  void clear() {
    xSemaphoreTake(this->mtx_, portMAX_DELAY);
    this->tail_ = this->head_;
    xSemaphoreGive(this->mtx_);
  }

 protected:
  void copy_in_(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) this->buf_[(this->head_ + i) & (N - 1)] = data[i];
    this->head_ += len;
  }
  uint8_t buf_[N];
  size_t head_{0}; /* free-running counters; head_ - tail_ is the fill level */
  size_t tail_{0};
  SemaphoreHandle_t mtx_{nullptr};
};

class TermowebRadio {
 public:
  /* Pin assignment, from the YAML's substitutions. Must be called before
   * start(); ignored afterwards (the SPI bus is already set up by then). */
  void set_pins(int sck, int mosi, int miso, int csn, int gdo0, int gdo2) {
    if (this->started_) return;
    this->pin_sck_ = (gpio_num_t) sck;
    this->pin_mosi_ = (gpio_num_t) mosi;
    this->pin_miso_ = (gpio_num_t) miso;
    this->pin_csn_ = (gpio_num_t) csn;
    this->pin_gdo0_ = (gpio_num_t) gdo0;
    this->pin_gdo2_ = (gpio_num_t) gdo2;
  }

  /* Cheap, thread-safe status for ESPHome sensors (read from the loop task). */
  bool client_connected() const { return this->client_connected_.load(); }
  uint32_t frames_received() const { return this->frames_ok_.load(); }
  std::string version() const { return TERMOWEB_RADIO_VERSION; }

  /* Called once from ESPHome's on_boot (priority -100, i.e. after WiFi and
   * the rest of the core are set up). Safe to call again: later calls are
   * no-ops. Everything that touches the CC1101 happens inside the radio
   * task, never here, so this returns immediately. */
  void start() {
    if (this->started_) return;
    this->started_ = true;
    uint8_t mac[6] = {0};
    if (esp_read_mac(mac, ESP_MAC_WIFI_STA) == ESP_OK)
      snprintf(this->mac_str_, sizeof(this->mac_str_), "%02X:%02X:%02X:%02X:%02X:%02X", mac[0], mac[1], mac[2],
               mac[3], mac[4], mac[5]);
    this->cmd_ring_.init();
    this->out_ring_.init();
    this->client_mtx_ = xSemaphoreCreateMutex();

    /* start() runs on ESPHome's loop task, so its core is the one to keep
     * the radio off. */
    BaseType_t loop_core = xPortGetCoreID();
    BaseType_t radio_core = portNUM_PROCESSORS > 1 ? (loop_core == 0 ? 1 : 0) : 0;
    /* Priority 19: above lwIP's tcpip thread (18) so network bursts cannot
     * delay an ack or let the 64-byte RX FIFO overflow, below the WiFi
     * driver (23) and esp_timer (22). The only long busy-wait it ever does
     * is a transmit (at most ~220 ms for a 255-byte frame, bounded just as
     * in cc1101.c), far inside the 5 s task watchdog. */
    xTaskCreatePinnedToCore(&TermowebRadio::radio_task_entry_, "tw_radio", 6144, this, 19, &this->radio_task_,
                            radio_core);
    xTaskCreatePinnedToCore(&TermowebRadio::writer_task_entry_, "tw_tcp_tx", 4096, this, 5, &this->writer_task_,
                            tskNO_AFFINITY);
    xTaskCreatePinnedToCore(&TermowebRadio::listener_task_entry_, "tw_tcp_rx", 4096, this, 5,
                            &this->listener_task_, tskNO_AFFINITY);
    ESP_LOGI(TAG, "started: radio task on core %d (ESPHome loop on core %d), TCP port %u", (int) radio_core,
             (int) loop_core, (unsigned) TCP_PORT);
  }

  /* Runtime carrier setting for YAML lambdas: becomes the frequency every
   * new connection starts at (the "power-on default" a DTR reset restores),
   * and retunes right away. Out-of-band values are refused. */
  bool set_frequency_khz(uint32_t khz) {
    if (khz < FREQ_KHZ_MIN || khz > FREQ_KHZ_MAX) return false;
    this->default_khz_.store(khz);
    if (this->radio_task_ != nullptr) xTaskNotify(this->radio_task_, EV_RETUNE, eSetBits);
    return true;
  }
  uint32_t frequency_khz() const { return this->freq_khz_.load(); }

  /* Feed command bytes to the parser exactly as if they had arrived on the
   * TCP connection (e.g. "D\n", "Q\n"), for diagnostics driven from ESPHome
   * itself (a lambda, a periodic 'D' status dump) with no host connected. Replies still go to the TCP client if there is one,
   * and always to the ESPHome log via emit_(). */
  void inject(const std::string &cmd) {
    if (cmd.empty()) return;
    if (this->radio_task_ == nullptr) return;
    this->push_cmd_((const uint8_t *) cmd.data(), cmd.size());
  }

  /* FREQ word for a carrier in kHz: round(f * 2^16 / 26 MHz). 869525 ->
   * 0x21717A, the cc1101.c value, so the default is bit-identical. */
  static uint32_t freq_word(uint32_t khz) { return (uint32_t) (((uint64_t) khz * 65536ULL + 13000ULL) / 26000ULL); }

 protected:
  enum RxState : uint8_t {
    RX_IDLE = 0,   /* nothing received since the last full frame */
    RX_LENPENDING, /* sync seen, waiting for byte 0 to learn the total */
    RX_FIXED,      /* total known, chip switched to fixed length mode */
    RX_DONE,       /* full frame + status bytes captured, packet handler owns rx_buf_ */
  };
  /* Incremental replacement for main.c's blocking reads after a command
   * letter: A = uart_getc_blocking, HEX_I/HEX_T = read_hex_line. The
   * ESP32-only commands reuse the same two shapes: Y like A (one digit),
   * HEX_N like I (a hex line, here exactly 2 bytes); DEC_F reads decimal. */
  enum ParseState : uint8_t { P_IDLE = 0, P_A, P_Y, P_HEX_I, P_HEX_N, P_HEX_T, P_DEC_F };

  /* ---------------------------------------------------------------- clock */

  /* main.c's micros(): a free-running 32-bit microsecond count, wrapping
   * every ~71.6 minutes exactly like the AVR's, so the printed field has
   * the same range and the same unsigned-subtraction semantics. */
  static uint32_t micros32() { return (uint32_t) esp_timer_get_time(); }

  /* ------------------------------------------------------------- SPI/chip */

  /* CC1101 datasheet 10.1: after CS goes low, wait for MISO (SO) low before
   * clocking, so the chip has left SLEEP/XOFF and its crystal is running.
   * cc1101.c spins forever; here the wait is bounded at 10 ms so a missing
   * or dead module costs log lines, not a hung task (and with it the TCP
   * replies). The MISO pad's input stays enabled while the SPI peripheral
   * owns it, so gpio_get_level reads the live level whether the pin is
   * routed through the GPIO matrix or the IO MUX. */
  void cs_low_() {
    gpio_set_level(this->pin_csn_, 0);
    int64_t t0 = esp_timer_get_time();
    while (gpio_get_level(this->pin_miso_)) {
      if (esp_timer_get_time() - t0 > 10000) {
        if ((this->miso_timeouts_++ % 100) == 0)
          ESP_LOGE(TAG, "CC1101 MISO stayed high after CS low (%u times): check wiring/power",
                   (unsigned) this->miso_timeouts_);
        return;
      }
    }
  }
  void cs_high_() { gpio_set_level(this->pin_csn_, 1); }

  /* One full-duplex transfer while CS is already held low by the caller.
   * Without DMA, a single spi_master transaction is capped at 64 bytes
   * (SOC_SPI_MAXIMUM_BUFFER_SIZE), so longer bursts are split into several
   * transactions; this is exactly why CS is a manual GPIO here, as on the
   * nanoCUL: the CC1101 ends a burst access as soon as CS rises, so CS must
   * stay low across every chunk of one burst. */
  void spi_xfer_(const uint8_t *tx, uint8_t *rx, size_t len) {
    static const uint8_t ZEROS[64] = {0};
    while (len > 0) {
      size_t chunk = len > 64 ? 64 : len;
      spi_transaction_t t;
      memset(&t, 0, sizeof(t));
      t.length = chunk * 8;
      t.tx_buffer = tx != nullptr ? tx : ZEROS;
      t.rx_buffer = rx;
      spi_device_polling_transmit(this->spi_, &t);
      if (tx != nullptr) tx += chunk;
      if (rx != nullptr) rx += chunk;
      len -= chunk;
    }
  }

  uint8_t strobe_(uint8_t cmd) {
    uint8_t status = 0;
    this->cs_low_();
    this->spi_xfer_(&cmd, &status, 1);
    this->cs_high_();
    return status;
  }
  void write_reg_(uint8_t addr, uint8_t val) {
    uint8_t b[2] = {addr, val};
    this->cs_low_();
    this->spi_xfer_(b, nullptr, 2);
    this->cs_high_();
  }
  void write_burst_(uint8_t addr, const uint8_t *buf, size_t len) {
    uint8_t hdr = addr | CC1101_WRITE_BURST;
    this->cs_low_();
    this->spi_xfer_(&hdr, nullptr, 1);
    this->spi_xfer_(buf, nullptr, len);
    this->cs_high_();
  }
  uint8_t read_status_(uint8_t addr) {
    uint8_t tx[2] = {(uint8_t) (addr | CC1101_READ_BURST), 0};
    uint8_t rx[2] = {0, 0};
    this->cs_low_();
    this->spi_xfer_(tx, rx, 2);
    this->cs_high_();
    return rx[1];
  }
  void read_burst_(uint8_t addr, uint8_t *buf, size_t len) {
    uint8_t hdr = addr | CC1101_READ_BURST;
    this->cs_low_();
    this->spi_xfer_(&hdr, nullptr, 1);
    this->spi_xfer_(nullptr, buf, len);
    this->cs_high_();
  }

  void write_freq_(uint32_t word) {
    this->write_reg_(CC1101_FREQ2, (uint8_t) (word >> 16));
    this->write_reg_(CC1101_FREQ1, (uint8_t) (word >> 8));
    this->write_reg_(CC1101_FREQ0, (uint8_t) word);
  }

  void write_sync_() {
    this->write_reg_(CC1101_SYNC1, SYNC1_VAL);
    this->write_reg_(CC1101_SYNC0, this->sync0_());
  }

  /* cc1101.c cc1101_init(), minus spi_init (done once in radio_setup_):
   * CS pulse, SRES, the register table, SCAL, PATABLE. FREQ and SYNC come
   * from the session state rather than the table. */
  void cc1101_init_() {
    this->cs_low_();
    this->cs_high_();
    esp_rom_delay_us(40);
    this->strobe_(CC1101_SRES);
    /* SRES restarts the crystal; the next cs_low_() waits for MISO low,
     * which is the chip saying it is ready again. */
    esp_rom_delay_us(2000);
    for (const auto &r : CONFIG) this->write_reg_(r.addr, r.val);
    this->write_freq_(freq_word(this->freq_khz_.load()));
    this->write_sync_();
    this->strobe_(CC1101_SCAL);
    esp_rom_delay_us(1000);
    uint8_t pa[2] = {PA_VAL, PA_VAL};
    this->write_burst_(CC1101_PATABLE, pa, 2);
  }

  /* cc1101.c: re-arm RX in infinite packet length mode for the next, as yet
   * unknown, frame length. */
  void enter_rx_() {
    this->strobe_(CC1101_SIDLE);
    this->strobe_(CC1101_SFRX);
    this->write_reg_(CC1101_PKTCTRL0, 0x02);
    this->strobe_(CC1101_SRX);
  }

  /* cc1101.c: switch to fixed length mid-reception; the chip's own byte
   * counter keeps running through the switch, so the packet ends exactly at
   * `total` and GDO0 falls there (see cc1101.h's comment). */
  void set_fixed_length_(uint8_t total) {
    this->write_reg_(CC1101_PKTLEN, total);
    this->write_reg_(CC1101_PKTCTRL0, 0x00);
  }

  uint8_t tx_recover_() {
    this->strobe_(CC1101_SFTX);
    this->enter_rx_();
    return CC1101_TX_UNDERFLOW;
  }

  /* cc1101.c cc1101_transmit(), 3.3 chunked version, unchanged in logic:
   * len <= 64 in one burst before STX; longer frames start on a 63-byte
   * fill and are topped up whenever TXBYTES drops under 32. Same poll
   * bounds (2000) and the same 100 us busy-wait, so the timing envelope
   * matches the AVR's; esp_rom_delay_us busy-waits, which is what keeps the
   * refill loop ahead of the 833 us/byte FIFO drain without depending on
   * the 1 ms FreeRTOS tick. Returns 0, CC1101_TX_UNDERFLOW, or the stuck
   * MARCSTATE. Callers bracket it with gdo_mask_()/gdo_unmask_(). */
  uint8_t transmit_(const uint8_t *buf, uint8_t len) {
    this->strobe_(CC1101_SIDLE);
    this->strobe_(CC1101_SFTX);
    /* RX may be in infinite-length mode; TX needs fixed length. */
    this->write_reg_(CC1101_PKTCTRL0, 0x00);
    this->write_reg_(CC1101_PKTLEN, len);

    if (len <= CC1101_TX_FIFO_SIZE) {
      this->write_burst_(CC1101_FIFO_ADDR, buf, len);
      this->strobe_(CC1101_STX);
    } else {
      this->write_burst_(CC1101_FIFO_ADDR, buf, CC1101_TX_INITIAL_FILL);
      this->strobe_(CC1101_STX);
      uint8_t sent = CC1101_TX_INITIAL_FILL;
      for (uint16_t poll = 0; sent < len; poll++) {
        if (poll >= 2000) {
          uint8_t stuck = this->read_status_(CC1101_MARCSTATE) & 0x1F;
          this->strobe_(CC1101_SFTX);
          this->enter_rx_();
          return stuck;
        }
        uint8_t txbytes = this->read_status_(CC1101_TXBYTES);
        if (txbytes & 0x80) return this->tx_recover_();
        uint8_t occupied = txbytes & 0x7F;
        if (occupied < CC1101_TX_REFILL_THRESHOLD) {
          /* -1 keeps clear of the completely-full TXBYTES ambiguity. */
          uint8_t space = CC1101_TX_FIFO_SIZE - occupied - 1;
          uint8_t remaining = len - sent;
          uint8_t chunk = remaining < space ? remaining : space;
          if (chunk > 0) {
            this->write_burst_(CC1101_FIFO_ADDR, buf + sent, chunk);
            sent += chunk;
          }
        }
        esp_rom_delay_us(100);
      }
    }

    uint8_t state = 0;
    for (uint16_t i = 0; i < 2000; i++) {
      esp_rom_delay_us(100);
      state = this->read_status_(CC1101_MARCSTATE) & 0x1F;
      if (state == 0x01) break;
      if (state == MARCSTATE_TXFIFO_UNDERFLOW) return this->tx_recover_();
    }
    this->enter_rx_();
    return state == 0x01 ? 0 : state;
  }

  /* ------------------------------------------------- GPIO ISRs / masking */

  /* main.c masks INT0/INT1 in EIMSK around every transmit and clears EIFR
   * afterwards, so edges caused by our own transmission (GDO0 also pulses
   * on a sent packet's sync/end) never count as receptions. Equivalent
   * here: while gdo_masked_ the ISRs return without doing anything (not
   * even counting syncs, as a masked INT1 would not), and unmasking clears
   * any GDO bit that was already latched in the task's notification value
   * (the EIFR clear). CMD/CONNECT/RETUNE bits are left alone. */
  void gdo_mask_() { this->gdo_masked_ = true; }
  void gdo_unmask_() {
    ulTaskNotifyValueClear(nullptr, EV_GDO0 | EV_GDO2);
    this->gdo_masked_ = false;
  }

  /* ISR(INT1_vect)'s cheap part: count the edge (main.c's sync_count) and
   * stamp the time. The stamp is taken here, at the true edge, and used as
   * rx_done_micros, so task scheduling latency never shows up in the RX
   * line's micros field (and therefore never in nanocul.py's
   * stick_delay_ms). */
  static void IRAM_ATTR gdo0_isr_(void *arg) {
    auto *self = static_cast<TermowebRadio *>(arg);
    if (self->gdo_masked_) return;
    self->sync_count_++;
    self->gdo0_edge_us_ = (uint32_t) esp_timer_get_time();
    BaseType_t woken = pdFALSE;
    xTaskNotifyFromISR(self->radio_task_, EV_GDO0, eSetBits, &woken);
    portYIELD_FROM_ISR(woken);
  }
  static void IRAM_ATTR gdo2_isr_(void *arg) {
    auto *self = static_cast<TermowebRadio *>(arg);
    if (self->gdo_masked_) return;
    BaseType_t woken = pdFALSE;
    xTaskNotifyFromISR(self->radio_task_, EV_GDO2, eSetBits, &woken);
    portYIELD_FROM_ISR(woken);
  }

  /* ---------------------------------------------------------- RX state */

  /* main.c rx_drain_fifo(), unchanged except for the dialect's length rule:
   * copy whatever the RX FIFO holds into rx_buf_, and once byte 0 is in,
   * compute the frame's total length from it and switch the chip into
   * fixed length mode mid-reception. A hardware overflow, or a total outside
   * [8, 255], flushes and re-arms rather than trusting the bytes. */
  void rx_drain_fifo_() {
    uint8_t status = this->read_status_(CC1101_RXBYTES);
    if (status & 0x80) {
      this->strobe_(CC1101_SIDLE);
      this->strobe_(CC1101_SFRX);
      this->enter_rx_();
      this->rx_have_ = 0;
      this->rx_state_ = RX_IDLE;
      return;
    }
    uint8_t n = status & 0x7F;
    if (n == 0) return;
    uint16_t room = RX_BUF_LEN - this->rx_have_;
    uint8_t to_read = (n > room) ? (uint8_t) room : n;
    if (to_read > 0) {
      this->read_burst_(CC1101_FIFO_ADDR, &this->rx_buf_[this->rx_have_], to_read);
      this->rx_have_ += to_read;
    }
    if (this->rx_state_ == RX_LENPENDING && this->rx_have_ >= 1) {
      /* Dialect A: byte 0 is (total - 3) scrambled by the keystream's own
       * first byte, 0xFF. Dialect B: byte 0 is the total, in the clear. */
      uint16_t total = this->dialect_ == DIALECT_B ? (uint16_t) this->rx_buf_[0]
                                                   : (uint16_t) ((this->rx_buf_[0] ^ 0xFF) + 3);
      if (total < RX_TOTAL_MIN || total > RX_TOTAL_MAX) {
        this->strobe_(CC1101_SIDLE);
        this->strobe_(CC1101_SFRX);
        this->enter_rx_();
        this->rx_have_ = 0;
        this->rx_state_ = RX_IDLE;
        return;
      }
      this->rx_total_ = total;
      this->set_fixed_length_((uint8_t) total);
      this->rx_state_ = RX_FIXED;
    }
  }

  /* ISR(INT0_vect)'s body: GDO2 rose, RX FIFO at/above the 4-byte
   * threshold. Repeats through a frame, draining it in small chunks so a
   * frame past 64 bytes never overflows the FIFO. Ignored while RX_DONE:
   * the previous frame still owns rx_buf_, and new bytes wait in the
   * hardware FIFO. */
  void on_gdo2_() {
    if (this->rx_state_ == RX_DONE) return;
    if (this->rx_state_ == RX_IDLE) {
      this->rx_have_ = 0;
      this->rx_state_ = RX_LENPENDING;
    }
    this->rx_drain_fifo_();
  }

  /* ISR(INT1_vect)'s body: GDO0 fell, end of packet once PKTLEN is in.
   * Collect the tail (including the 2 APPEND_STATUS bytes), hand the frame
   * to the packet handler, and re-arm RX unconditionally. */
  void on_gdo0_(uint32_t edge_us) {
    if (this->rx_state_ == RX_FIXED || this->rx_state_ == RX_LENPENDING) {
      this->rx_drain_fifo_();
      this->rx_done_micros_ = edge_us;
      this->rx_state_ = RX_DONE;
      this->packet_ready_ = true;
    }
    this->enter_rx_();
  }

  /* -------------------------------------------------------------- output */

  /* Queue one line for the client with main.c's "\r\n" terminator, and log
   * it: traffic (RX/TX/ACK) at DEBUG, status lines at INFO, TXERR at WARN.
   * Never blocks: if the output ring cannot hold the whole line (a client
   * that stopped reading), the line is dropped whole and counted. */
  void emit_(const char *line) {
    /* Braces matter: below DEBUG log level ESP_LOGD expands to nothing. */
    if (line[0] == '#') {
      ESP_LOGI(TAG, "%s", line);
    } else if (strncmp(line, "TXERR", 5) == 0) {
      ESP_LOGW(TAG, "%s", line);
    } else {
      ESP_LOGD(TAG, "%s", line);
    }
    size_t len = strlen(line);
    char buf[LINE_MAX_ + 2];
    if (len > LINE_MAX_) len = LINE_MAX_;
    memcpy(buf, line, len);
    buf[len++] = '\r';
    buf[len++] = '\n';
    if (!this->out_ring_.push_all((const uint8_t *) buf, len)) {
      if ((this->out_dropped_++ % 50) == 0)
        ESP_LOGW(TAG, "client not reading: dropped %u output line(s)", (unsigned) this->out_dropped_);
      return;
    }
    if (this->writer_task_ != nullptr) xTaskNotifyGive(this->writer_task_);
  }

  /* Append helpers for building one line in a fixed buffer. */
  __attribute__((format(printf, 3, 4))) static void append_(char *buf, size_t &pos, const char *fmt, ...) {
    if (pos >= LINE_MAX_) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + pos, LINE_MAX_ + 1 - pos, fmt, ap);
    va_end(ap);
    if (n > 0) pos = (pos + (size_t) n > LINE_MAX_) ? LINE_MAX_ : pos + (size_t) n;
  }
  static void append_hex_(char *buf, size_t &pos, const uint8_t *data, size_t len) {
    static const char HEX[] = "0123456789ABCDEF";
    for (size_t i = 0; i < len && pos + 2 <= LINE_MAX_; i++) {
      buf[pos++] = HEX[data[i] >> 4];
      buf[pos++] = HEX[data[i] & 0x0F];
    }
    buf[pos] = '\0';
  }

  /* "freq=869.525": the banner's literal, generated from the current kHz so
   * an 'F' retune is visible in V/Q without any format change. */
  void freq_str_(char *out, size_t size) {
    uint32_t khz = this->freq_khz_.load();
    snprintf(out, size, "%u.%03u", (unsigned) (khz / 1000), (unsigned) (khz % 1000));
  }

  uint8_t sync0_() const { return this->dialect_ == DIALECT_B ? SYNC0_B : SYNC0_A; }

  /* termoweb_rx's banner and Q line, field for field, with the new fields
   * only APPENDED at the end so any parser of the old lines still matches:
   * mac= on both, dialect= and net= on Q. sync= reports the sync word the
   * chip is actually using for the current dialect. */
  void print_banner_() {
    char f[16], line[160];
    this->freq_str_(f, sizeof(f));
    snprintf(line, sizeof(line), "# termoweb_rx %s freq=%s rate=9.6k sync=%02X%02X mode=dynamic tx=pa%02X mac=%s",
             TERMOWEB_RADIO_VERSION, f, SYNC1_VAL, this->sync0_(), PA_VAL, this->mac_str_);
    this->emit_(line);
  }
  void print_query_() {
    char f[16], line[224];
    this->freq_str_(f, sizeof(f));
    snprintf(line, sizeof(line),
             "# Q termoweb_rx %s freq=%s pa=%02X sync=%02X%02X mode=dynamic autoack=%s id=%02X dialect=%c net=%04X "
             "mac=%s",
             TERMOWEB_RADIO_VERSION, f, PA_VAL, SYNC1_VAL, this->sync0_(), this->auto_ack_ ? "on" : "off",
             this->our_id_, this->dialect_ == DIALECT_B ? 'B' : 'A', (unsigned) this->net_, this->mac_str_);
    this->emit_(line);
  }

  /* ------------------------------------------------------------ CRCs */

  /* CRC-16/CCITT, poly 0x1021, init 0x1D0F, final XOR 0xFFFF (main.c,
   * _vendored_frame.crc16): dialect A's CRC, over descrambled bytes. */
  static uint16_t crc16_(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0x1D0F;
    for (uint16_t i = 0; i < len; i++) {
      crc ^= (uint16_t) data[i] << 8;
      for (uint8_t b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t) ((crc << 1) ^ 0x1021) : (uint16_t) (crc << 1);
    }
    return crc ^ 0xFFFF;
  }
  /* CRC-16/MODBUS: reflected poly 0xA001, init 0xFFFF, no final XOR;
   * dialect B's CRC, over the clear bytes. */
  static uint16_t crc_modbus_(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
      crc ^= data[i];
      for (uint8_t b = 0; b < 8; b++) crc = (crc & 1) ? (uint16_t) ((crc >> 1) ^ 0xA001) : (uint16_t) (crc >> 1);
    }
    return crc;
  }

  /* Does this on-air frame of `total` bytes carry a valid CRC in dialect
   * `d`? Only the verdict is computed; the frame itself is never altered,
   * since the host gets the bytes exactly as received. The length byte is
   * checked too, for symmetry with the host's own parse (it always matches
   * here, because the firmware ended the frame at the length that byte
   * gave). Dialect A descrambles into a local copy first, because its CRC is
   * over the logical bytes. */
  static bool frame_crc_ok_(Dialect d, const uint8_t *air, uint16_t total) {
    if (total < 5 || total > RX_TOTAL_MAX) return false;
    uint16_t crc;
    uint8_t hi, lo;
    if (d == DIALECT_B) {
      if (air[0] != total) return false;
      crc = crc_modbus_(air, total - 2);
      hi = air[total - 2];
      lo = air[total - 1];
    } else {
      uint8_t lg[RX_TOTAL_MAX];
      for (uint16_t i = 0; i < total; i++) lg[i] = air[i] ^ KEYSTREAM[i];
      if (lg[0] != total - 3) return false;
      crc = crc16_(lg, total - 2);
      hi = lg[total - 2];
      lo = lg[total - 1];
    }
    return hi == (uint8_t) (crc >> 8) && lo == (uint8_t) crc;
  }

  /* Addressing byte `i` (3 = src, 4 = dst, 5 = flags) of an on-air frame:
   * descrambled in A, already clear in B. */
  static uint8_t field_(Dialect d, const uint8_t *air, int i) {
    return d == DIALECT_B ? air[i] : (uint8_t) (air[i] ^ KEYSTREAM[i]);
  }

  /* ---------------------------------------------------------------- ack */

  /* The 8-byte link ack, on-air bytes, in dialect `d`, network `net`:
   *   A (main.c build_ack, _vendored_frame.build_ack): logical 05 <net>
   *     <acker> <acked sender> 80, CCITT CRC, all XORed with the keystream;
   *   B: 08 <net> <acker> <acked sender> 80, MODBUS CRC, in the clear.
   * Built natively in the firmware, not by the host, because a heater's ack
   * window (about 16 ms observed) is far shorter than a host round trip. */
  static void build_ack_(Dialect d, uint16_t net, uint8_t acker_id, uint8_t src_id, uint8_t out[8]) {
    uint8_t f[8] = {0x05, (uint8_t) (net >> 8), (uint8_t) net, acker_id, src_id, 0x80, 0, 0};
    if (d == DIALECT_B) {
      f[0] = 0x08;
      uint16_t crc = crc_modbus_(f, 6);
      f[6] = (uint8_t) (crc >> 8);
      f[7] = (uint8_t) crc;
      memcpy(out, f, 8);
    } else {
      uint16_t crc = crc16_(f, 6);
      f[6] = (uint8_t) (crc >> 8);
      f[7] = (uint8_t) crc;
      for (uint8_t i = 0; i < 8; i++) out[i] = f[i] ^ KEYSTREAM[i];
    }
  }

  /* Boot-time check of both dialects' ack builders and CRC checkers against
   * fixed vectors computed independently in Python (_vendored_frame.py's
   * build_ack/build_frame for A, a reference MODBUS for B), so this tests the
   * C against the host's own codec rather than against itself:
   *  - ack 01 -> 06: A, net 1B30 = FA9C8858B12189B6; B, net 1234 (synthetic) = 081234010680A0E4;
   *  - both acks, and a B registration frame (synthetic network id 1234) and the A poll
   *    build_frame(1, 6, 57 55), must pass frame_crc_ok_ in their own
   *    dialect and fail it in the other one (so a wrong 'Y' is visible as
   *    crc_ok=0, not silently accepted).
   * Only logs; a failure means acks and crc_ok verdicts are wrong, which is
   * worth an ERROR line but not worth refusing to run the receiver. */
  static bool self_test_() {
    static const uint8_t ACK_A[8] = {0xFA, 0x9C, 0x88, 0x58, 0xB1, 0x21, 0x89, 0xB6};
    static const uint8_t ACK_B[8] = {0x08, 0x12, 0x34, 0x01, 0x06, 0x80, 0xA0, 0xE4};
    static const uint8_t REG_B[15] = {0x0F, 0x12, 0x34, 0x06, 0x01, 0x00, 0x06, 0x01,
                                      0x01, 0x01, 0x01, 0x00, 0x50, 0x72, 0x22};
    static const uint8_t POLL_A[16] = {0xF2, 0x9C, 0x88, 0x58, 0xB1, 0xA1, 0xCD, 0x22,
                                       0x57, 0x5E, 0x4B, 0x9C, 0x59, 0xBC, 0x49, 0xC9};
    uint8_t a[8], b[8];
    build_ack_(DIALECT_A, 0x1B30, 0x01, 0x06, a);
    build_ack_(DIALECT_B, 0x1234, 0x01, 0x06, b);
    bool ack_a = memcmp(a, ACK_A, 8) == 0;
    bool ack_b = memcmp(b, ACK_B, 8) == 0;
    bool crc_a = frame_crc_ok_(DIALECT_A, ACK_A, 8) && frame_crc_ok_(DIALECT_A, POLL_A, 16) &&
                 !frame_crc_ok_(DIALECT_B, POLL_A, 16);
    bool crc_b = frame_crc_ok_(DIALECT_B, ACK_B, 8) && frame_crc_ok_(DIALECT_B, REG_B, 15) &&
                 !frame_crc_ok_(DIALECT_A, REG_B, 15);
    ESP_LOGI(TAG, "self-test dialect A: ack %s, CRC %s", ack_a ? "ok" : "FAIL", crc_a ? "ok" : "FAIL");
    ESP_LOGI(TAG, "self-test dialect B: ack %s, CRC %s", ack_b ? "ok" : "FAIL", crc_b ? "ok" : "FAIL");
    bool ok = ack_a && ack_b && crc_a && crc_b;
    if (ok)
      ESP_LOGI(TAG, "self-test passed");
    else
      ESP_LOGE(TAG, "self-test FAILED: auto-acks and crc_ok verdicts are not trustworthy");
    return ok;
  }

  /* ----------------------------------------------------- packet handler */

  /* main.c's `if (packet_ready)` block, in the same order: ack first, from
   * rx_buf_ directly, so it goes on air before any output work; then
   * snapshot, release the receiver, catch up on the FIFO, and only then
   * format and queue the lines.
   *
   * Two differences from termoweb_rx, both because this firmware now knows
   * the dialect's CRC: the RX line's crc_ok field is that real verdict
   * (termoweb_rx printed the CC1101's own CRC_OK bit, meaningless with
   * hardware CRC off), and auto-ack only fires for a frame whose CRC holds
   * (termoweb_rx acked on address alone; acking a corrupt frame would tell a
   * heater that a damaged command arrived). The frame bytes themselves are
   * printed exactly as received, CRC-valid or not, for the host to judge. */
  void handle_packet_() {
    this->packet_ready_ = false;
    uint16_t total = this->rx_total_;
    uint16_t have = this->rx_have_;
    uint32_t t = this->rx_done_micros_;
    Dialect d = this->dialect_;
    bool ack_sent = false;
    uint8_t ack[8];

    bool crc_ok = have >= total && frame_crc_ok_(d, this->rx_buf_, total);
    if (crc_ok) this->frames_ok_.fetch_add(1);

    if (this->auto_ack_ && crc_ok) {
      uint8_t src = field_(d, this->rx_buf_, 3);
      uint8_t dst = field_(d, this->rx_buf_, 4);
      uint8_t flags = field_(d, this->rx_buf_, 5);
      if (dst == this->our_id_ && flags == 0x00) {
        build_ack_(d, this->net_, this->our_id_, src, ack);
        this->gdo_mask_();
        uint8_t err = this->transmit_(ack, 8);
        this->gdo_unmask_();
        ack_sent = (err == 0);
      }
    }

    uint16_t local_have = have > RX_BUF_LEN ? RX_BUF_LEN : have;
    memcpy(this->scratch_, this->rx_buf_, local_have);

    /* Release rx_buf_ for the next frame, then drain what the RX_DONE gate
     * held back. As in main.c, this drain runs in RX_IDLE, so anything it
     * reads is discarded by the next GDO2 (which resets rx_have_ to 0). */
    this->rx_have_ = 0;
    this->rx_state_ = RX_IDLE;
    this->rx_drain_fifo_();

    char line[LINE_MAX_ + 1];
    size_t pos = 0;
    if (ack_sent) {
      /* main.c stamps the ACK line with micros() at print time, i.e. right
       * after the release/drain above; same here. */
      append_(line, pos, "ACK %lu ", (unsigned long) micros32());
      append_hex_(line, pos, ack, 8);
      this->emit_(line);
    }

    uint8_t rssi_raw = 0, lqi_raw = 0;
    if (local_have >= (uint16_t) total + 2) {
      rssi_raw = this->scratch_[total];
      lqi_raw = this->scratch_[total + 1];
    }
    int8_t rssi_signed = (int8_t) rssi_raw;
    int16_t rssi_dbm10 = (int16_t) rssi_signed * 5 - 740; /* dBm x10: raw/2 - 74 */
    int16_t rssi_mag = rssi_dbm10 < 0 ? -rssi_dbm10 : rssi_dbm10;
    uint8_t lqi = lqi_raw & 0x7F;
    uint8_t frame_len = (uint8_t) (total > 255 ? 255 : total);
    if (frame_len > local_have) frame_len = (uint8_t) local_have; /* truncated read: report what we have */

    pos = 0;
    append_(line, pos, "RX %lu %s%d.%d %u %u ", (unsigned long) t, rssi_dbm10 < 0 ? "-" : "", rssi_mag / 10,
            rssi_mag % 10, (unsigned) lqi, (unsigned) crc_ok);
    append_hex_(line, pos, this->scratch_, frame_len);
    if (this->raw_mode_) append_(line, pos, " %02X%02X", rssi_raw, lqi_raw);
    this->emit_(line);
  }

  /* --------------------------------------------------------- commands */

  static int8_t hex_nibble_(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  }

  /* 'T' after its hex line ended (n = read_hex_line's return value). The
   * bytes go on air exactly as given: building, scrambling and CRCing a
   * frame for the current dialect is the host codec's job. The 3.5 handler
   * order is kept: transmit with GDO masked, then re-arm and reset the RX
   * state machine BEFORE unmasking, so bytes of any frame discarded by the
   * transmit are flushed before RX_IDLE lets a new frame start (main.c's
   * long 3.5 comment). */
  void finish_t_(uint8_t n) {
    char line[LINE_MAX_ + 1];
    size_t pos = 0;
    if (n == 0) {
      this->emit_("TXERR empty or bad hex");
      return;
    }
    this->gdo_mask_();
    uint8_t err = this->transmit_(this->tx_buf_, n);
    this->enter_rx_();
    this->rx_have_ = 0;
    this->rx_state_ = RX_IDLE;
    this->packet_ready_ = false;
    this->gdo_unmask_();
    if (err == CC1101_TX_UNDERFLOW) {
      this->emit_("TXERR underflow");
    } else if (err) {
      append_(line, pos, "TXERR marcstate=%02X", err);
      this->emit_(line);
    } else {
      append_(line, pos, "TX %lu %u ", (unsigned long) micros32(), (unsigned) n);
      append_hex_(line, pos, this->tx_buf_, n);
      this->emit_(line);
    }
  }

  void finish_i_(uint8_t n) {
    char line[48];
    if (n == 1) {
      this->our_id_ = this->id_buf_[0];
      snprintf(line, sizeof(line), "# id=%02X", this->our_id_);
      this->emit_(line);
    } else {
      this->emit_("# I? expected 2 hex digits");
    }
  }

  /* ESP32-only 'N<hex4>': the network id auto-acks carry. Only the ack
   * builder uses it; received frames are never filtered on it, so the host
   * still sees every network's traffic. Session state, back to 1B30 on the
   * next connection. */
  void finish_n_(uint8_t n) {
    char line[48];
    if (n == 2) {
      this->net_ = (uint16_t) ((this->id_buf_[0] << 8) | this->id_buf_[1]);
      snprintf(line, sizeof(line), "# net=%04X", (unsigned) this->net_);
      this->emit_(line);
    } else {
      this->emit_("# N? expected 4 hex digits");
    }
  }

  /* ESP32-only 'Y0'/'Y1': switch dialect, which changes the sync word, the
   * length rule, the CRC verdict and the ack format all at once. The sync
   * word is rewritten live with the same masked re-arm pattern as 'T' and
   * 'F': SIDLE (SYNC registers may change in any state, but a frame half
   * received under the old sync word would be ended by the wrong length
   * rule), write SYNC1/SYNC0, then enter_rx_ flushes the RX FIFO and
   * re-arms, and the RX state machine restarts from RX_IDLE. */
  void set_dialect_(Dialect d) {
    this->dialect_ = d;
    this->gdo_mask_();
    this->strobe_(CC1101_SIDLE);
    this->write_sync_();
    this->enter_rx_();
    this->rx_have_ = 0;
    this->rx_state_ = RX_IDLE;
    this->packet_ready_ = false;
    this->gdo_unmask_();
    char line[48];
    snprintf(line, sizeof(line), "# dialect=%c sync=%02X%02X", d == DIALECT_B ? 'B' : 'A', SYNC1_VAL, this->sync0_());
    this->emit_(line);
  }

  /* ESP32-only 'F<kHz>': same masked re-arm pattern as 'T', since retuning
   * from IDLE discards any frame in flight. SIDLE first because FREQ
   * registers must not be written while the synthesiser is running; the
   * SRX inside enter_rx_ recalibrates (MCSM0 autocal on IDLE->RX). */
  void retune_(uint32_t khz) {
    this->freq_khz_.store(khz);
    this->gdo_mask_();
    this->strobe_(CC1101_SIDLE);
    this->write_freq_(freq_word(khz));
    this->enter_rx_();
    this->rx_have_ = 0;
    this->rx_state_ = RX_IDLE;
    this->packet_ready_ = false;
    this->gdo_unmask_();
  }
  void finish_f_(bool ok) {
    if (!ok || this->f_digits_ == 0 || this->f_value_ < FREQ_KHZ_MIN || this->f_value_ > FREQ_KHZ_MAX) {
      this->emit_("# F? expected kHz");
      return;
    }
    this->retune_(this->f_value_);
    char f[16], line[64];
    this->freq_str_(f, sizeof(f));
    snprintf(line, sizeof(line), "# freq=%s word=%06X", f, (unsigned) freq_word(this->f_value_));
    this->emit_(line);
  }

  /* End a pending multi-byte command with read_hex_line's/
   * uart_getc_blocking's failure value (timeout, bad byte, overflow). */
  void parse_fail_() {
    ParseState s = this->pstate_;
    this->pstate_ = P_IDLE;
    switch (s) {
      case P_A:
        this->emit_("# A? expected 0 or 1");
        break;
      case P_Y:
        this->emit_("# Y? expected 0 or 1");
        break;
      case P_HEX_I:
        this->finish_i_(0);
        break;
      case P_HEX_N:
        this->finish_n_(0);
        break;
      case P_HEX_T:
        this->finish_t_(0);
        break;
      case P_DEC_F:
        this->finish_f_(false);
        break;
      default:
        break;
    }
  }

  /* One byte of input, as main.c's main loop plus the blocking helpers
   * would consume it. Returns true when the byte completed a command that
   * touched the radio, so the caller can service pending GDO events before
   * parsing more. */
  bool feed_(char c) {
    switch (this->pstate_) {
      case P_IDLE:
        break;
      case P_A:
        this->pstate_ = P_IDLE;
        if (c == '0' || c == '1') {
          this->auto_ack_ = (c == '1');
          this->emit_(this->auto_ack_ ? "# autoack=on" : "# autoack=off");
        } else {
          this->emit_("# A? expected 0 or 1");
        }
        return false;
      case P_Y:
        this->pstate_ = P_IDLE;
        if (c == '0' || c == '1') {
          this->set_dialect_(c == '1' ? DIALECT_B : DIALECT_A);
          return true;
        }
        this->emit_("# Y? expected 0 or 1");
        return false;
      case P_HEX_I:
      case P_HEX_N:
      case P_HEX_T: {
        /* read_hex_line(): newline ends the line (success only on an even
         * nibble count); any other non-hex byte, or a byte past `max`,
         * fails it with that byte consumed. */
        uint8_t *out = this->pstate_ == P_HEX_T ? this->tx_buf_ : this->id_buf_;
        uint16_t max = this->pstate_ == P_HEX_T ? TX_MAX_LEN : (this->pstate_ == P_HEX_N ? 2 : 1);
        bool is_t = this->pstate_ == P_HEX_T;
        if (c == '\n' || c == '\r') {
          uint8_t n = this->hi_ < 0 ? this->hex_n_ : 0;
          ParseState done = this->pstate_;
          this->pstate_ = P_IDLE;
          if (is_t) {
            this->finish_t_(n);
            return true;
          }
          if (done == P_HEX_N)
            this->finish_n_(n);
          else
            this->finish_i_(n);
          return false;
        }
        int8_t v = hex_nibble_(c);
        if (v < 0) {
          this->parse_fail_();
          return is_t;
        }
        if (this->hi_ < 0) {
          this->hi_ = v;
        } else {
          if (this->hex_n_ >= max) {
            this->parse_fail_();
            return is_t;
          }
          out[this->hex_n_++] = (uint8_t) (this->hi_ << 4) | (uint8_t) v;
          this->hi_ = -1;
        }
        return false;
      }
      case P_DEC_F:
        if (c == '\n' || c == '\r') {
          this->pstate_ = P_IDLE;
          this->finish_f_(true);
          return true;
        }
        if (c >= '0' && c <= '9' && this->f_digits_ < 7) {
          this->f_value_ = this->f_value_ * 10 + (uint32_t) (c - '0');
          this->f_digits_++;
          return false;
        }
        this->parse_fail_();
        return false;
    }

    /* P_IDLE: main.c's command dispatch. Unknown bytes (including the '\n'
     * that ends every command line) are ignored, as on the nanoCUL. */
    this->cmd_start_us_ = micros32();
    switch (c) {
      case 'V':
        this->print_banner_();
        break;
      case 'Q':
        this->print_query_();
        break;
      case 'X':
        this->raw_mode_ = !this->raw_mode_;
        this->emit_(this->raw_mode_ ? "# raw=on" : "# raw=off");
        break;
      case 'A':
        this->pstate_ = P_A;
        break;
      case 'Y':
        this->pstate_ = P_Y;
        break;
      case 'I':
      case 'N':
      case 'T':
        this->pstate_ = c == 'T' ? P_HEX_T : (c == 'N' ? P_HEX_N : P_HEX_I);
        this->hi_ = -1;
        this->hex_n_ = 0;
        break;
      case 'F':
        this->pstate_ = P_DEC_F;
        this->f_value_ = 0;
        this->f_digits_ = 0;
        break;
      case 'D': {
        /* On-demand chip state dump. */
        uint8_t marcstate = this->read_status_(CC1101_MARCSTATE);
        uint8_t pktstatus = this->read_status_(CC1101_PKTSTATUS);
        uint8_t rxbytes = this->read_status_(CC1101_RXBYTES);
        char line[128];
        snprintf(line, sizeof(line), "# marcstate=%02X pktstatus=%02X rxbytes=%02X syncs=%u rxreset=%u", marcstate,
                 pktstatus, rxbytes, (unsigned) this->sync_count_, (unsigned) this->rxreset_count_);
        this->emit_(line);
        break;
      }
      default:
        break;
    }
    return false;
  }

  /* ------------------------------------------------------- radio task */

  /* Everything a DTR reset would put back: session state to power-on
   * defaults, the chip re-initialised, the banner printed, RX armed. Runs
   * at boot and on every new TCP connection. GDO is masked throughout, as
   * main.c keeps interrupts off until cc1101_init()/enter_rx() are done
   * (the 3.2.1 fix: GDO2 toggles CHIP_RDYn during SRES). */
  void reset_session_() {
    this->gdo_mask_();
    this->cmd_ring_.clear();
    this->pstate_ = P_IDLE;
    this->raw_mode_ = false;
    this->auto_ack_ = false;
    this->dialect_ = DIALECT_A;
    this->net_ = DEFAULT_NET;
    this->our_id_ = 0x01;
    this->sync_count_ = 0;
    this->rxreset_count_ = 0;
    this->rx_have_ = 0;
    this->rx_total_ = 0;
    this->rx_state_ = RX_IDLE;
    this->packet_ready_ = false;
    this->freq_khz_.store(this->default_khz_.load());
    this->cc1101_init_();
    this->print_banner_();
    this->enter_rx_();
    this->gdo_unmask_();
  }

  bool radio_setup_() {
    gpio_config_t out = {};
    out.pin_bit_mask = 1ULL << this->pin_csn_;
    out.mode = GPIO_MODE_OUTPUT;
    gpio_config(&out);
    gpio_set_level(this->pin_csn_, 1);

    spi_bus_config_t bus = {};
    bus.mosi_io_num = this->pin_mosi_;
    bus.miso_io_num = this->pin_miso_;
    bus.sclk_io_num = this->pin_sck_;
    bus.quadwp_io_num = -1;
    bus.quadhd_io_num = -1;
    bus.max_transfer_sz = 64;
    esp_err_t err = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_DISABLED);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "spi_bus_initialize: %s", esp_err_to_name(err));
      return false;
    }
    spi_device_interface_config_t dev = {};
    dev.mode = 0;
    /* 4 MHz, the nanoCUL's own SPI clock (fosc/4): comfortably inside the
     * CC1101's 6.5 MHz burst-access limit without needing its inter-byte
     * delays. */
    dev.clock_speed_hz = 4000000;
    dev.spics_io_num = -1; /* manual CS, see spi_xfer_ */
    dev.queue_size = 1;
    err = spi_bus_add_device(SPI2_HOST, &dev, &this->spi_);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "spi_bus_add_device: %s", esp_err_to_name(err));
      return false;
    }
    /* Only this task ever uses the bus; holding it permanently makes every
     * polling transaction skip the bus arbitration. */
    spi_device_acquire_bus(this->spi_, portMAX_DELAY);

    gpio_config_t in = {};
    in.pin_bit_mask = (1ULL << this->pin_gdo0_) | (1ULL << this->pin_gdo2_);
    in.mode = GPIO_MODE_INPUT;
    in.intr_type = GPIO_INTR_DISABLE;
    gpio_config(&in);

    /* The ack/CRC check needs no hardware; run it first so its verdict is
     * in the log even if the CC1101 turns out to be missing. */
    self_test_();
    ESP_LOGI(TAG, "pins SCK=%d MOSI=%d MISO=%d CSN=%d GDO0=%d GDO2=%d, MAC %s", (int) this->pin_sck_,
             (int) this->pin_mosi_, (int) this->pin_miso_, (int) this->pin_csn_, (int) this->pin_gdo0_,
             (int) this->pin_gdo2_, this->mac_str_);

    this->gdo_mask_();
    this->cc1101_init_();
    uint8_t partnum = this->read_status_(CC1101_PARTNUM);
    uint8_t version = this->read_status_(CC1101_VERSION);
    if (partnum == 0x00 && version == 0x14)
      ESP_LOGI(TAG, "CC1101 found: partnum=%02X version=%02X", partnum, version);
    else
      ESP_LOGE(TAG, "unexpected CC1101 id partnum=%02X version=%02X (expected 00/14)", partnum, version);

    /* ESPHome may already have installed the shared GPIO ISR service (any
     * component using pin interrupts does); that is ESP_ERR_INVALID_STATE
     * and fine, the handlers below attach to it either way. */
    err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
      ESP_LOGE(TAG, "gpio_install_isr_service: %s", esp_err_to_name(err));
      return false;
    }
    gpio_set_intr_type(this->pin_gdo0_, GPIO_INTR_NEGEDGE); /* packet done, as main.c's INT1 */
    gpio_set_intr_type(this->pin_gdo2_, GPIO_INTR_POSEDGE); /* FIFO threshold, as main.c's INT0 */
    gpio_isr_handler_add(this->pin_gdo0_, &TermowebRadio::gdo0_isr_, this);
    gpio_isr_handler_add(this->pin_gdo2_, &TermowebRadio::gdo2_isr_, this);
    gpio_intr_enable(this->pin_gdo0_);
    gpio_intr_enable(this->pin_gdo2_);
    return true;
  }

  static void radio_task_entry_(void *arg) { static_cast<TermowebRadio *>(arg)->radio_loop_(); }

  void radio_loop_() {
    if (!this->radio_setup_()) {
      ESP_LOGE(TAG, "radio setup failed; radio task stopped");
      vTaskDelete(nullptr);
      return;
    }
    this->reset_session_(); /* boot: banner to the log, RX armed */
    ESP_LOGI(TAG, "receiver armed at %u kHz", (unsigned) this->freq_khz_.load());

    for (;;) {
      /* Sleep until an edge, a command byte, a connection or a retune, but
       * never more than 50 ms, so the 1 s command timeouts and the 250 ms
       * liveness guard are checked on time; don't sleep at all if command
       * bytes are already waiting (an earlier iteration stopped parsing
       * after a 'T' to service the radio first). */
      uint32_t bits = 0;
      TickType_t wait = this->cmd_ring_.empty() ? pdMS_TO_TICKS(50) : 0;
      xTaskNotifyWait(0, UINT32_MAX, &bits, wait);

      if (bits & EV_CONNECT) {
        this->reset_session_();
        xTaskNotifyGive(this->listener_task_); /* listener may now read the new socket */
      }
      if (bits & EV_RETUNE) {
        this->retune_(this->default_khz_.load());
        ESP_LOGI(TAG, "retuned to %u kHz (word %06X)", (unsigned) this->freq_khz_.load(),
                 (unsigned) freq_word(this->freq_khz_.load()));
      }
      /* GDO2 before GDO0: within one frame the threshold edge always comes
       * first, so if task latency let both latch, this is their real order. */
      if (bits & EV_GDO2) this->on_gdo2_();
      if (bits & EV_GDO0) this->on_gdo0_(this->gdo0_edge_us_);

      if (this->packet_ready_) this->handle_packet_();

      if (this->cmd_overrun_.exchange(false)) this->emit_("# uart overrun");

      /* Commands: parse until the input is empty, or until a command that
       * used the radio finishes, so GDO events queued meanwhile are handled
       * before the next one. */
      uint8_t c;
      bool any = false;
      while (this->cmd_ring_.pop(&c, 1) == 1) {
        any = true;
        if (this->feed_((char) c)) break;
      }
      /* Command timeouts are judged only when no byte is waiting, exactly
       * like read_hex_line/uart_getc_blocking, which only look at the clock
       * when the UART is empty. */
      if (!any && this->pstate_ != P_IDLE && (micros32() - this->cmd_start_us_) > CMD_TIMEOUT_US)
        this->parse_fail_();

      /* main.c 3.5 liveness guard: rx_state stuck at RX_DONE with no packet
       * pending. In this port the GDO0 handler and the packet handler run
       * back to back in this one task, so the deadlock it was written for
       * cannot form; it is kept, with main.c's 250 ms threshold and its
       * "# rxreset n=" report, so any future path with the same mistake
       * still recovers loudly instead of going silent. */
      if (this->rx_state_ == RX_DONE && !this->packet_ready_ &&
          (micros32() - this->rx_done_micros_) > RXRESET_TIMEOUT_US) {
        this->enter_rx_();
        this->rx_have_ = 0;
        this->rx_state_ = RX_IDLE;
        this->rxreset_count_++;
        char line[48];
        snprintf(line, sizeof(line), "# rxreset n=%u", (unsigned) this->rxreset_count_);
        this->emit_(line);
      }
    }
  }

  /* --------------------------------------------------------- TCP tasks */

  static void writer_task_entry_(void *arg) { static_cast<TermowebRadio *>(arg)->writer_loop_(); }

  /* Drains the output ring into the current client. Holds client_mtx_
   * across pop + send so a connection swap can never slip in between and
   * deliver the old session's bytes to the new client ahead of its banner.
   * With no client the bytes are discarded, like UART output with nothing
   * on the other end. A send error only shuts the socket down; the listener
   * notices (recv returns 0) and does the close, so only one task ever
   * closes a client fd. */
  void writer_loop_() {
    static uint8_t buf[1024];
    for (;;) {
      ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
      for (;;) {
        xSemaphoreTake(this->client_mtx_, portMAX_DELAY);
        size_t n = this->out_ring_.pop(buf, sizeof(buf));
        if (n == 0) {
          xSemaphoreGive(this->client_mtx_);
          break;
        }
        int fd = this->client_fd_;
        size_t off = 0;
        while (fd >= 0 && off < n) {
          int r = ::send(fd, buf + off, n - off, 0);
          if (r <= 0) {
            ESP_LOGW(TAG, "send failed (errno %d); dropping client", errno);
            ::shutdown(fd, SHUT_RDWR);
            break;
          }
          off += (size_t) r;
        }
        xSemaphoreGive(this->client_mtx_);
      }
    }
  }

  static void listener_task_entry_(void *arg) { static_cast<TermowebRadio *>(arg)->listener_loop_(); }

  int open_listener_() {
    int fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) return -1;
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0 || ::listen(fd, 2) != 0) {
      ESP_LOGW(TAG, "bind/listen on port %u failed (errno %d), retrying", (unsigned) TCP_PORT, errno);
      ::close(fd);
      return -1;
    }
    ESP_LOGI(TAG, "listening on TCP port %u", (unsigned) TCP_PORT);
    return fd;
  }

  /* A new client replaces the old one, as a second process opening a
   * serial port would take it over. Socket options:
   *  - TCP_NODELAY: commands and reply lines are tiny and latency-bound
   *    (nanocul.py waits 160 ms for an ack); Nagle plus the peer's delayed
   *    ACK would add up to ~200 ms per line.
   *  - keepalive (30 s idle, 3 x 10 s probes): a host that vanished
   *    without a FIN (power loss, WiFi drop) is detected within ~1 min
   *    even if nobody connects to replace it.
   *  - SO_SNDTIMEO 1 s: a stalled client can hold the writer, and with it
   *    a connection swap, for at most that long. */
  void accept_client_(int lfd) {
    struct sockaddr_in peer = {};
    socklen_t plen = sizeof(peer);
    int fd = ::accept(lfd, (struct sockaddr *) &peer, &plen);
    if (fd < 0) return;
    int one = 1, idle = 30, intvl = 10, cnt = 3;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
    struct timeval tv = {1, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    xSemaphoreTake(this->client_mtx_, portMAX_DELAY);
    int old = this->client_fd_;
    this->client_fd_ = fd;
    this->client_connected_.store(true);
    this->out_ring_.clear(); /* whatever the old session had not sent yet dies with it */
    xSemaphoreGive(this->client_mtx_);
    if (old >= 0) {
      ::shutdown(old, SHUT_RDWR);
      ::close(old);
      ESP_LOGI(TAG, "previous client replaced");
    }
    char ip[16];
    inet_ntoa_r(peer.sin_addr, ip, sizeof(ip));
    ESP_LOGI(TAG, "client connected from %s", ip);

    /* The DTR-reset equivalent: the radio task re-inits and prints the
     * banner. Wait for it before reading this socket, so none of the new
     * client's bytes can be swept away by that reset's command-ring clear
     * (nanocul.py sleeps 2.5 s before its first write anyway). */
    xTaskNotify(this->radio_task_, EV_CONNECT, eSetBits);
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(2000));
  }

  void drop_client_(int fd, const char *why) {
    xSemaphoreTake(this->client_mtx_, portMAX_DELAY);
    if (this->client_fd_ == fd) {
      this->client_fd_ = -1;
      this->client_connected_.store(false);
    }
    xSemaphoreGive(this->client_mtx_);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    ESP_LOGI(TAG, "client disconnected (%s)", why);
  }

  /* Bytes from the client into the command ring, the TCP side of main.c's
   * USART_RX_vect ring. TCP already has flow control, so a full ring is
   * first given up to ~100 ms to drain; only then is the rest dropped and
   * reported as "# uart overrun", the line main.c prints for a full ring. */
  void push_cmd_(const uint8_t *data, size_t len) {
    for (int tries = 0; len > 0; tries++) {
      size_t n = this->cmd_ring_.push_some(data, len);
      data += n;
      len -= n;
      xTaskNotify(this->radio_task_, EV_CMD, eSetBits);
      if (len == 0) return;
      if (tries >= 100) {
        this->cmd_overrun_.store(true);
        return;
      }
      vTaskDelay(pdMS_TO_TICKS(1));
    }
  }

  void listener_loop_() {
    static uint8_t buf[512];
    int lfd = -1;
    for (;;) {
      if (lfd < 0) {
        lfd = this->open_listener_();
        if (lfd < 0) {
          vTaskDelay(pdMS_TO_TICKS(1000));
          continue;
        }
      }
      /* Only this task writes client_fd_, so reading it unlocked is safe. */
      int cfd = this->client_fd_;
      fd_set rfds;
      FD_ZERO(&rfds);
      FD_SET(lfd, &rfds);
      int maxfd = lfd;
      if (cfd >= 0) {
        FD_SET(cfd, &rfds);
        if (cfd > maxfd) maxfd = cfd;
      }
      struct timeval tv = {1, 0};
      int r = ::select(maxfd + 1, &rfds, nullptr, nullptr, &tv);
      if (r < 0) {
        ESP_LOGW(TAG, "select failed (errno %d)", errno);
        vTaskDelay(pdMS_TO_TICKS(100));
        continue;
      }
      if (r == 0) continue;
      if (FD_ISSET(lfd, &rfds)) {
        /* A replaced client's unread bytes are abandoned with it. */
        this->accept_client_(lfd);
        continue;
      }
      if (cfd >= 0 && FD_ISSET(cfd, &rfds)) {
        int n = ::recv(cfd, buf, sizeof(buf), 0);
        if (n <= 0) {
          this->drop_client_(cfd, n == 0 ? "closed by peer" : "recv error");
          continue;
        }
        this->push_cmd_(buf, (size_t) n);
      }
    }
  }

  /* -------------------------------------------------------------- state */

  /* Longest line: "RX " + 10-digit micros + " -128.0 127 1 " + 255 bytes
   * as hex + " XXXX" raw status = 541 characters. */
  static constexpr size_t LINE_MAX_ = 600;

  bool started_{false};
  gpio_num_t pin_sck_{(gpio_num_t) DEFAULT_PIN_SCK};
  gpio_num_t pin_mosi_{(gpio_num_t) DEFAULT_PIN_MOSI};
  gpio_num_t pin_miso_{(gpio_num_t) DEFAULT_PIN_MISO};
  gpio_num_t pin_csn_{(gpio_num_t) DEFAULT_PIN_CSN};
  gpio_num_t pin_gdo0_{(gpio_num_t) DEFAULT_PIN_GDO0};
  gpio_num_t pin_gdo2_{(gpio_num_t) DEFAULT_PIN_GDO2};
  char mac_str_[18]{"00:00:00:00:00:00"}; /* WiFi STA MAC, read once in start() */
  std::atomic<uint32_t> frames_ok_{0};      /* CRC-valid frames since boot, for a YAML sensor */
  std::atomic<bool> client_connected_{false};
  TaskHandle_t radio_task_{nullptr};
  TaskHandle_t writer_task_{nullptr};
  TaskHandle_t listener_task_{nullptr};
  spi_device_handle_t spi_{nullptr};

  /* Written by the GPIO ISRs, read by the radio task. */
  volatile bool gdo_masked_{true};
  volatile uint16_t sync_count_{0}; /* 'D' syncs=: GDO0 ISR firings incl. noise, uint16 like main.c */
  volatile uint32_t gdo0_edge_us_{0};

  /* Radio task only (main.c's rx_* globals, minus the volatile they needed
   * for ISR sharing on the AVR). */
  uint8_t rx_buf_[RX_BUF_LEN];
  uint8_t scratch_[RX_BUF_LEN]; /* RX snapshot; TX payload has its own tx_buf_ here */
  uint8_t tx_buf_[TX_MAX_LEN];
  uint8_t id_buf_[2]; /* 'I' (1 byte) and 'N' (2 bytes) */
  uint16_t rx_have_{0};
  uint16_t rx_total_{0};
  RxState rx_state_{RX_IDLE};
  uint32_t rx_done_micros_{0};
  bool packet_ready_{false};
  uint16_t rxreset_count_{0};
  uint32_t miso_timeouts_{0};
  uint32_t out_dropped_{0};

  /* Session state, reset on every connection. */
  bool auto_ack_{false};
  Dialect dialect_{DIALECT_A}; /* 'Y'; read by the length rule, CRC verdict and ack builder */
  uint16_t net_{DEFAULT_NET};  /* 'N'; network id in auto-acks */
  uint8_t our_id_{0x01};
  bool raw_mode_{false};
  ParseState pstate_{P_IDLE};
  uint32_t cmd_start_us_{0};
  int8_t hi_{-1};
  uint16_t hex_n_{0};
  uint32_t f_value_{0};
  uint8_t f_digits_{0};

  std::atomic<uint32_t> freq_khz_{DEFAULT_FREQ_KHZ};
  std::atomic<uint32_t> default_khz_{DEFAULT_FREQ_KHZ};
  std::atomic<bool> cmd_overrun_{false};

  /* TCP side. */
  SemaphoreHandle_t client_mtx_{nullptr};
  int client_fd_{-1};
  ByteRing<1024> cmd_ring_;
  ByteRing<8192> out_ring_;
};

/* The one instance; ESPHome YAML calls termoweb::radio.start() from on_boot. */
inline TermowebRadio radio;

}  // namespace termoweb

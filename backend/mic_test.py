"""
Standalone microphone sanity test.

Run this FIRST to verify your mic is working before launching the voice loop.
Prints continuous energy readings with a visual bar so you can see your
speech levels in real time.

Usage:
    python mic_test.py
    python mic_test.py --device 9     # specific device index
"""

import sys
import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_SIZE = 320  # 20ms at 16kHz
BAR_WIDTH = 60

def main() -> None:
    device = None
    for arg in sys.argv[1:]:
        if arg == "--device":
            continue
        try:
            device = int(arg)
        except ValueError:
            pass
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device = int(sys.argv[idx + 1])

    print("=" * 60)
    print("MICROPHONE SANITY TEST")
    print("=" * 60)
    print()
    print("Available devices:")
    print(sd.query_devices())
    print()
    print(f"Using device: {device if device is not None else 'system default'}")
    print(f"Sample rate: {SAMPLE_RATE}, Block size: {BLOCK_SIZE} ({BLOCK_SIZE * 1000 // SAMPLE_RATE}ms)")
    print()
    print("Speak into your microphone. You should see the bar grow.")
    print("Press Ctrl+C to stop.")
    print()
    print(f"{'Energy':>8}  {'':—<{BAR_WIDTH}}  Status")

    frame_count = 0
    energies = []

    def callback(indata, frames, time_info, status):
        nonlocal frame_count
        if status:
            print(f"  [WARNING] {status}")

        audio = indata.flatten().astype(np.float32)
        energy = float(np.abs(audio).mean())
        energies.append(energy)
        frame_count += 1

        # Print every 5 frames (~100ms) to avoid flooding
        if frame_count % 5 == 0:
            avg_energy = np.mean(energies[-5:])
            # Scale bar: map 0-500 energy to 0-BAR_WIDTH chars
            bar_len = min(BAR_WIDTH, int(avg_energy / 500 * BAR_WIDTH))
            bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
            status = "SPEECH" if avg_energy > 10 else "silence"
            print(f"\r{avg_energy:8.1f}  {bar}  {status:8s}", end="", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="int16",
            device=device,
            callback=callback,
        ):
            print()  # newline before streaming starts
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        print()
        if energies:
            arr = np.array(energies)
            print(f"Captured {len(energies)} frames ({len(energies) * 20 / 1000:.1f}s)")
            print(f"Energy — min: {arr.min():.1f}, max: {arr.max():.1f}, "
                  f"mean: {arr.mean():.1f}, p95: {np.percentile(arr, 95):.1f}")
            print()
            if arr.max() < 5:
                print("WARNING: Max energy is very low. Your mic may be muted or gain is too low.")
            elif arr.max() < 50:
                print("NOTE: Energy levels are low but detectable. Speak louder or move closer to the mic.")
            else:
                print("OK: Speech energy detected at normal levels.")
    except sd.PortAudioError as e:
        print(f"ERROR: Could not open audio device: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

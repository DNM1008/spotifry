import sys
import json
import numpy as np
import essentia
import essentia.standard as es


def extract_features(audio_file):
    # Load audio
    loader = es.MonoLoader(filename=audio_file)()

    # Frame generator
    frame_size = 1024
    hop_size = 512
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC()

    # Collect statistics
    mfccs = []

    for frame in essentia.standard.FrameGenerator(
        loader, frameSize=frame_size, hopSize=hop_size, startFromZero=True
    ):
        spec = spectrum(window(frame))
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)

    mfccs = np.array(mfccs)

    features = {
        "mean_mfcc": np.mean(mfccs, axis=0).tolist(),
        "var_mfcc": np.var(mfccs, axis=0).tolist(),
        "rms": float(np.sqrt(np.mean(loader**2))),
        "zero_crossing_rate": float(
            ((loader[:-1] * loader[1:]) < 0).sum() / len(loader)
        ),
    }

    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <audiofile>")
        sys.exit(1)

    audio_file = sys.argv[1]
    features = extract_features(audio_file)
    print(json.dumps(features, indent=2))

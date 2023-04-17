import argparse
from test2 import *

def load_model(model_path, input_size, num_classes):
    model = TransformerAudioClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_audio(audio_path, n_mels):
    audio, _ = librosa.load(audio_path, sr=None, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=audio, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = torch.tensor(log_mel_spec).permute(1, 0)
    return log_mel_spec.unsqueeze(0)

def classify_audio(model, audio_path, device, n_mels=128):
    log_mel_spec = preprocess_audio(audio_path, n_mels)
    log_mel_spec = log_mel_spec.to(device)
    with torch.no_grad():
        output = model(log_mel_spec)
    _, predicted_label = torch.max(output, dim=1)
    return predicted_label.item()

def main():
    parser = argparse.ArgumentParser(description="Classify audio using a trained Transformer model")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--audio_path", type=str, default="C:/Users/User/Desktop/130.도시 소리 데이터/01.데이터/2.Validation/원천데이터/1.자동차/2.차량사이렌/1.자동차_293_1.wav")
    args = parser.parse_args()

    input_size = 128
    num_classes = 209
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.model_path, input_size, num_classes).to(device)
    predicted_label = classify_audio(model, args.audio_path, device)
    print(f"Predicted label for '{args.audio_path}': {predicted_label}")

if __name__ == "__main__":
    main()
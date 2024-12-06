import unittest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from predict import (
    preprocess_audio_chunk,
    predict_speaker_from_chunk,
    predict_speaker_from_file,
    continuous_record_and_predict,
)

class TestSpeakerRecognition(unittest.TestCase):
    def setUp(self):
        # Create a dummy model and label encoder for testing
        self.mock_model = MagicMock()
        self.mock_label_encoder = MagicMock()

        # Patch the model and label encoder loading
        self.model_patch = patch('main_script.model', self.mock_model)
        self.encoder_patch = patch('main_script.label_encoder', self.mock_label_encoder)
        
        self.model_patch.start()
        self.encoder_patch.start()
        
        # Mock model predictions
        self.mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])  # Mock softmax output
        self.mock_label_encoder.inverse_transform.return_value = ["Speaker1"]

    def tearDown(self):
        self.model_patch.stop()
        self.encoder_patch.stop()

    def test_preprocess_audio_chunk(self):
        # Test preprocessing of an audio chunk
        audio_chunk = np.random.rand(44100)  # Simulated 1-second audio chunk
        mel_spectrogram = preprocess_audio_chunk(audio_chunk, sr=44100, augment=False)
        self.assertEqual(mel_spectrogram.shape, (128, 300, 1))  # Check the shape

    def test_predict_speaker_from_chunk(self):
        # Test prediction from an audio chunk
        audio_chunk = np.random.rand(44100)  # Simulated 1-second audio chunk
        speaker, confidence = predict_speaker_from_chunk(audio_chunk, sr=44100)
        self.assertEqual(speaker, "Speaker1")
        self.assertAlmostEqual(confidence, 0.7, places=2)

    def test_predict_speaker_from_file(self):
        # Test prediction from an audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio_file:
            # Simulate the process_single_audio_file function
            with patch('main_script.process_single_audio_file') as mock_process:
                mock_process.return_value = np.random.rand(128, 300)  # Simulated mel spectrogram
                predict_speaker_from_file(tmp_audio_file.name, augment=False)
                mock_process.assert_called_once_with(tmp_audio_file.name, max_length=300, augment=False)

    @patch('main_script.sd.rec', return_value=np.random.rand(44100, 1))
    @patch('main_script.wavio.write')
    def test_continuous_record_and_predict(self, mock_write, mock_rec):
        # Test continuous recording and prediction
        with patch('builtins.input', side_effect=["2"]):  # Simulate user choice
            continuous_record_and_predict(duration=1, confidence_threshold=0.5)
            mock_rec.assert_called()
            mock_write.assert_called()
            self.mock_model.predict.assert_called()

if __name__ == "__main__":
    unittest.main()
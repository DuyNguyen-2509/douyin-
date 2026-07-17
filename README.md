# Douyin Translator with Logo Blur

Tự động tải video Douyin, dịch và lồng tiếng sang nhiều ngôn ngữ, đồng thời làm mờ logo trên video.

## Yêu cầu hệ thống
- Python 3.10 – 3.12
- FFmpeg (có trong PATH)
- Kết nối Internet (để tải mô hình Whisper, dịch Google, TTS Edge, tải video)
- (Tùy chọn) CUDA 11.8+ cho GPU

## Cài đặt
```bash
git clone https://github.com/yourusername/douyin-translator.git
cd douyin-translator
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# EduBridge

EduBridge là prototype tạo học liệu tạm thời từ các tài liệu giáo dục mở có license rõ ràng.

Pipeline:

```text
OER → Chunking → Multilingual Embedding → Chroma Retrieval → Gemini → Learning Pack
```

Prototype hiện hỗ trợ:
- Hàm số bậc nhất
- Định lý Pythagoras
- Xác suất cơ bản

Demo: https://pglppswthwdaf7qpgwdapp4.streamlit.app/

Nguồn và license được lấy trực tiếp từ metadata của tài liệu đã retrieve, không do LLM tự tạo.

# ISS_Object_Detect_WebApp

Full-stack HTML-Docker-Python ISS WebApp

Place .pth model into "Model" directory:
https://drive.google.com/file/d/1tRq_BUI9jIKqKexg-BI2Lh1A_6h6D39E/view?usp=sharing

For program specs and details see: https://github.com/dsrushton/ISS_Object_Detect/blob/main/README.md

Use: 
build -t iss-detection .   
docker run --gpus all -p 5000:5000 -v "${PWD}/model:/app/model" iss-detection

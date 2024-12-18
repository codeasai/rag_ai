import os
import sys
import subprocess
from pathlib import Path

def install_yaml():
    """ติดตั้ง PyYAML"""
    try:
        print("กำลังติดตั้ง PyYAML...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML==6.0.1"])
        global yaml
        import yaml
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการติดตั้ง PyYAML: {str(e)}")
        return False

def check_python_version():
    """ตรวจสอบเวอร์ชัน Python"""
    if sys.version_info < (3, 8):
        raise SystemError("ต้องการ Python 3.8 หรือใหม่กว่า")
    return True

def install_requirements():
    """ติดตั้ง dependencies"""
    try:
        # ตรวจสอบว่ามีไฟล์ requirements.txt
        if not Path("requirements.txt").exists():
            print("สร้างไฟล์ requirements.txt...")
            requirements = [
                "langchain>=0.1.12",
                "langchain-community>=0.0.28",
                "transformers>=4.38.2",
                "torch>=2.2.1",
                "faiss-cpu>=1.8.0",
                "scikit-learn>=1.4.1.post1",
                "tqdm>=4.66.2",
                "pyyaml>=6.0.1",
                "sentence-transformers>=2.5.1",
                "pypdf>=4.0.1"
            ]
            
            with open("requirements.txt", "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(requirements))
        
        # อัปเกรด pip ก่อน
        print("\nกำลังอัปเกรด pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # ติดตั้ง pypdf ก่อน
        print("\nกำลังติดตั้ง pypdf...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf==4.0.1"])
        except subprocess.CalledProcessError as e:
            print(f"ไม่สามารถติดตั้ง pypdf: {str(e)}")
            return False
        
        # ติดตั้ง dependencies ทั้งหมด
        print("\nกำลังติดตั้ง dependencies ทั้งหมด...")
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                "requirements.txt",
                "--verbose"
            ])
            print("\nติดตั้ง dependencies เสร็จสมบูรณ์")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nเกิดข้อผิดพลาดในการติดตั้ง dependencies: {str(e)}")
            return False
            
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดในการติดตั้ง: {str(e)}")
        return False

def create_directory_structure():
    """สร้างโครงสร้างโฟลเดอร์ที่จำเป็น"""
    directories = [
        "src",
        "data/raw_pdfs",
        "data/processed",
        "models",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return True

def create_config_file():
    """สร้างไฟล์ config.yaml ถ้ายังไม่มี"""
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        default_config = {
            "data": {
                "raw_dir": "data/raw_pdfs",
                "processed_dir": "data/processed"
            },
            "model": {
                "name": "bert-base-multilingual-cased",
                "batch_size": 8,
                "epochs": 3
            },
            "training": {
                "learning_rate": 2e-5
            },
            "model_dir": "models",
            "log_dir": "logs"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    return True

def verify_setup():
    """ตรวจสอบว่าทุกอย่างพร้อมใช้งาน"""
    required_files = [
        "config.yaml",
        "requirements.txt",
        "src/prepare_rag.py",
        "src/train.py",
        "src/evaluate.py"
    ]
    
    required_dirs = [
        "src",
        "data/raw_pdfs",
        "data/processed",
        "models",
        "logs"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    
    if missing_files:
        print(f"ไม่พบไฟล์ที่จำเป็น: {', '.join(missing_files)}")
        return False
        
    if missing_dirs:
        print(f"ไม่พบโฟลเดอร์ที่จำเป็น: {', '.join(missing_dirs)}")
        return False
        
    return True

def main():
    print("กำลังตรวจสอบและเตรียมระบบ...")
    
    try:
        # ตรวจสอบเวอร์ชัน Python
        if not check_python_version():
            return
        
        # สิดตั้ง PyYAML ก่อน
        if not install_yaml():
            return
        
        # สร้างโครงสร้างโฟลเดอร์
        print("กำลังสร้างโครงสร้างโฟลเดอร์...")
        if not create_directory_structure():
            return
            
        # ติดตั้ง dependencies
        print("กำลังติดตั้ง dependencies...")
        if not install_requirements():
            return
            
        # สร้างไฟล์ config
        print("กำลังตรวจสอบและสร้างไฟล์ config...")
        if not create_config_file():
            return
            
        # ตรวจสอบการติดตั้ง
        if verify_setup():
            print("""
การเตรียมระบบเสร็จสมบูรณ์ คุณสามารถเริ่มใช้งานโปรแกรมได้โดย:
1. ใส่ไฟล์ PDF ที่ต้องการประมวลผลนี้ลงใน data/raw_pdfs
2. รันคำสั่งต่อไปนี้ตามลำดับ:
   python src/prepare_rag.py
   python src/train.py
   python src/evaluate.py
            """)
        else:
            print("พบข้อผิดพลาดในการเตรียมระบบ กรุณาตรวจสอบข้อความด้านบน")
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main() 
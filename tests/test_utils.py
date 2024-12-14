import unittest
import tempfile
import os
from pathlib import Path
import yaml
import sys

# เพิ่ม project root ใน Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config, setup_logging, create_directories


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'data': {
                'pdf_dir': str(Path(self.temp_dir) / 'data/raw_pdfs'),
                'processed_dir': str(Path(self.temp_dir) / 'data/processed'),
            },
            'model': {
                'name': 'bert-base-uncased',
                'epochs': 1
            },
            'model_dir': str(Path(self.temp_dir) / 'models'),
            'log_dir': str(Path(self.temp_dir) / 'logs')
        }

        self.config_path = str(Path(self.temp_dir) / 'config.yaml')
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # บันทึก config ด้วย encoding ที่ถูกต้อง
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f)

    def test_load_config(self):
        loaded_config = load_config(self.config_path)
        self.assertEqual(loaded_config, self.config)

    def test_setup_logging(self):
        log_dir = str(Path(self.temp_dir) / 'logs')
        setup_logging(log_dir)
        self.assertTrue(Path(log_dir).exists())

        import logging
        logger = logging.getLogger('test')
        logger.info('Test message')

        log_file = Path(log_dir) / 'training.log'
        self.assertTrue(log_file.exists())

    def test_create_directories(self):
        create_directories(self.config)
        paths_to_check = [
            Path(self.config['data']['pdf_dir']),
            Path(self.config['data']['processed_dir']),
            Path(self.config['model_dir']),
            Path(self.config['log_dir'])
        ]

        for path in paths_to_check:
            self.assertTrue(path.exists())

    def test_basic_functionality(self):
        # ทดสอบว่าสามารถสร้างและเข้าถึงไดเรกทอรีได้
        test_dir = Path(self.temp_dir) / 'test_basic'
        test_dir.mkdir(exist_ok=True)
        
        # เขียนไฟล์ทดสอบ
        test_file = test_dir / 'test.txt'
        test_file.write_text('test content')
        
        # ทดสอบการอ่านไฟล์
        self.assertEqual(test_file.read_text(), 'test content')
        self.assertTrue(test_dir.exists())

    def test_config_format(self):
        # ทดสอบว่า config มี format ถูกต้อง
        required_keys = ['data', 'model', 'model_dir', 'log_dir']
        loaded_config = load_config(self.config_path)
        
        for key in required_keys:
            self.assertIn(key, loaded_config)

    def test_create_directories_permission_error(self):
        # ทดสอบกรณีไม่มีสิทธิ์สร้างไดเรกทอรี
        import stat
        read_only_dir = Path(self.temp_dir) / 'readonly'
        read_only_dir.mkdir()
        os.chmod(read_only_dir, stat.S_IREAD)

        config_with_readonly = self.config.copy()
        config_with_readonly['data']['pdf_dir'] = str(read_only_dir / 'test')
        
        with self.assertRaises(PermissionError):
            create_directories(config_with_readonly)

    def test_load_config_invalid_path(self):
        # ทดสอบกรณีไฟล์ config ไม่มีอยู่จริง
        with self.assertRaises(FileNotFoundError):
            load_config('invalid/path/config.yaml')

    def test_load_config_invalid_format(self):
        # ทดสอบกรณี config format ไม่ถูกต้อง
        invalid_config_path = str(Path(self.temp_dir) / 'invalid_config.yaml')
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: format:')
        
        with self.assertRaises(yaml.YAMLError):
            load_config(invalid_config_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)

    
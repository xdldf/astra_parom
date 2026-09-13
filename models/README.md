# Комплектные модели

Эта папка хранится в Git вместе с исходниками. Общий размер — около 21 МБ. Git LFS не требуется.

| Файл | Назначение |
| --- | --- |
| `workbench/yolo26n.pt` | Обнаружение автомобилей, автобусов, грузовиков и мотоциклов |
| `plate-models/detector/yolo-v9-t-640-license-plate-end2end/yolo-v9-t-640-license-plates-end2end.onnx` | Обнаружение номерного знака |
| `plate-models/ocr/cct-s-v1-global-model/cct_s_v1_global.onnx` | Чтение символов номера |
| `plate-models/ocr/cct-s-v1-global-model/cct_s_v1_global_plate_config.yaml` | Конфигурация модели чтения |

`INSTALL.cmd` проверяет SHA-256 по `manifest.json` и копирует отсутствующие файлы в соответствующие подпапки `web_app/data`. Существующие рабочие модели сохраняются. При повреждении комплектного файла установка завершается с указанием ошибки. Библиотеки устанавливаются через интернет, но веса отдельно скачивать не нужно.

В комплект включены используемые приложением модели; экспериментальные YOLO26x и CCT v2 не включены.

Проекты авторов моделей: [Ultralytics](https://github.com/ultralytics/ultralytics), [Open Image Models](https://github.com/ankandrew/open-image-models), [Fast Plate OCR](https://github.com/ankandrew/fast-plate-ocr). Модели сторонние; условия их использования определяются соответствующими авторами.

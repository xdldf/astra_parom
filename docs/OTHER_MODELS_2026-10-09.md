# Другие модели: проверка 9 октября 2026

Повторно запущен неизменённый профиль на десяти моделях-кандидатах, которых не было в прежних 78 каталожных примерах и семи последующих каталожных сравнениях. **Сохранены 10/10 оценок и полных исправленных кадров. Из восьми условно сопоставимых наблюдений пять укладываются в 10 см для всех перечисленных заводских вариантов, два не укладываются, одно зависит от версии.** Ещё два оставлены только справочно. Требование «менее 10 см для каждой машины» не выполнено.

Это новые **каталожные сравнения**, а не новые записи или впервые измеренные машины: все десять кадров уже встречались в общем техническом аудите 162 обнаружений. Использованы четыре из прежних шести локальных ST-видео. Скачать новую запись ST_2026-09-08_11-30-01.mp4 через доступный браузер не удалось, поэтому проверки на новом видео здесь нет. Коэффициенты и рабочая конфигурация не изменялись.

## Результаты

Модели определены по изображению предварительно; год, рынок и внешнее оборудование конкретных экземпляров не подтверждены. Расхождение ниже — сравнение с перечисленными каталожными вариантами при условии верной идентификации и штатного кузова, **не измеренная физическая погрешность**. Значения показаны округлённо; категории рассчитаны по исходным числам со строгим порогом `< 0.1 м`.

| Наблюдение / модель-кандидат | Оценка, м | Каталог, м | Расхождение, см | Результат |
|---|---:|---:|---:|---|
| V1-013 Suzuki Jimny JB23 | 3.593 | 3.395 | 19.8 | Более 10 см |
| V1-047 Nissan Serena C26 | 4.793 | 4.685–4.770 | 2.3–10.8 | Зависит от версии |
| V1-049 Nissan NV200 M20 / Mitsubishi Delica D:3 candidate | 4.445 | 4.400–4.410 | 3.5–4.5 | Менее 10 см |
| V1-054 Toyota Isis | 4.652 | 4.610–4.640 | 1.2–4.2 | Менее 10 см |
| V2-018 Toyota Sienta XP80 | 4.101 | 4.100 | 0.1 | Менее 10 см |
| V2-052 Renault Captur I / Kaptur | 4.343 | 4.122–4.333 | 1.0–22.1 | Только справочно |
| V4-015 Chery Tiggo 7 Pro | 4.464 | 4.500 | 3.6 | Менее 10 см |
| V4-019 UAZ Patriot | 4.571 | 4.750–4.785 | 17.9–21.4 | Более 10 см |
| V6-010 Mitsubishi L200 fourth generation, double cab | 5.200 | 5.000–5.185 | 1.5–20.0 | Только справочно |
| V6-048 Chevrolet Niva / Lada Niva, pre-Travel body | 4.006 | 4.056 | 5.0 | Менее 10 см |

Renault Captur и российский Kaptur похожи, но имеют разные длины; модель не установлена однозначно. У L200 не подтверждены рынок, длина платформы и бамперы. Их диапазоны служат контекстом и не входят в число 5/8. У Serena сохранены стандартные версии и Highway Star; для стандартных 4,685 м расхождение составляет 10,75 см, для 4,770 м — 2,25 см. Интервалы не являются полным перечнем всех возможных переделок.

## Что показали промахи

- **Jimny:** оценка 3,593 м против штатных 3,395 м, расхождение +19,84 см. Контур вышел за обучающий диапазон (`outside_feature_range`); число выдала резервная модель по контуру и колёсам. Обычная временная оценка 3,805 м тоже завышена. Резервное число здесь не подтверждает точность.
- **Patriot:** оценка 4,571 м против 4,750–4,785 м, расхождение −17,91…−21,41 см. Контур и основная модель колёс допустили оценку. Контур отдельно даёт 4,593 м, временной расчёт — 4,559 м. Согласие разных оценок в этом случае не означает близости к каталогу; сам по себе контроль стабильности такой промах не исключает.

Оба кадра находятся у измерительной линии. Эти промахи нельзя объяснить только пропуском центрирования из-за задержки. Причина конкретного смещения по одному кадру не установлена; результаты демонстрируют ограничение обобщения текущей калибровки на другие формы кузова. Все оценки сохраняют `accuracy_validated: false`.

## Источники размеров

- **V1-013 Suzuki Jimny JB23**: [Suzuki JB23 official history](https://www.suzuki.co.jp/car/jimny/special/history/jb23.html); [Suzuki 2010 JB23 brochure](https://www.suzuki.co.jp/car/jimny/special/history/catalog/JB23.pdf).
- **V1-047 Nissan Serena C26**: [Nissan C26 specifications](https://history.nissan.co.jp/SERENA/C26/1312/PDF/serena_specification.pdf).
- **V1-049 Nissan NV200 M20 / Mitsubishi Delica D:3 candidate**: [Nissan NV200 official specifications](https://www-asia.nissan-cdn.net/content/dam/Nissan/jp/vehicles/nv200vanette/2407/pdf/nv200vanette_specsheet.pdf).
- **V1-054 Toyota Isis**: [Toyota vehicle history: Isis](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60000003/index.html).
- **V2-018 Toyota Sienta XP80**: [Toyota Sienta X 2006 catalogue](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-SIENTA/200605/10033764/).
- **V2-052 Renault Captur I / Kaptur**: [Renault Captur 2013 press release](https://presse.renault.de/renault-captur-vereint-vorteile-von-suv-und-kompaktlimousine-1/?lang=deu); [Renault Kaptur official brochure](https://cdn.group.renault.com/ren/ru/new-kaptur/renault_KAPTUR_10_09.pdf.asset.pdf/25c119e468.pdf).
- **V4-015 Chery Tiggo 7 Pro**: [Chery Kazakhstan Tiggo 7 Pro specifications](https://chery.kz/userdata/rubrics/rubrics_117/file_ru.pdf?1748847938=).
- **V4-019 UAZ Patriot**: [UAZ Patriot official brochure, 2022](https://www.uaz.ru/data/uploads/uaz/originals/uaz-patriot-catalog-011122.pdf).
- **V6-010 Mitsubishi L200 fourth generation, double cab**: [L200 type approval ROSС TH.MT02.E05539, rehosted original](https://ruscoc.ru/ru/home/mitsubishi-ka0t-l200-sertifikat-pocc-tn-mt02-e05539/%D0%A0%D0%9E%D0%A1%D0%A1%20%D0%A2%D0%9D.%D0%9C%D0%A202.%D0%9505539-original.pdf); [Mitsubishi UK 2014 brochure, rehosted manufacturer document](https://autocatalogarchive.com/wp-content/uploads/2017/05/Mitsubishi-L200-2014-UK.pdf).
- **V6-048 Chevrolet Niva / Lada Niva, pre-Travel body**: [Chevrolet Niva manufacturer specification sheet, dealer-hosted](https://niva-ufa.ru/upload/iblock/ab1/ab11b7a7c19135b23393c9d75c5f21f6.pdf).

Примечания к источникам: для Jimny используется стандартный JB23, без предположения об исключении запаски из 3,395 м. У Patriot завод указывает 4,750/4,785 м **без контейнера / с контейнером запасного колеса**, а не без самой запаски. Для Niva берётся 4,056 м со штатной запаской; 3,919 м относится к кузову без её выступа. Для NV200 приведены заводские размеры Nissan M20, при этом значок на машине не подтверждён. PDF L200 2014 UK доступен в поисковой выдержке, прямое скачивание не удалось; эта строка оставлена справочной. Подробные оговорки идентификации сохранены в JSON.

## Воспроизведение и сохранённые данные

[Выбор кадров](other-models-selection-2026-10-09.json) и [каталожные интервалы](other-models-catalogue-2026-10-09.json) зафиксированы до просмотра результатов повторного прогона. Каталог не передавался в расчёт. [Компактный отчёт](other-models-audit-2026-10-09.json) содержит исходные оценки, категории, источники и контрольные суммы видео, моделей, кода, кадров и записей станции. Все каталожные строки имеют `fit_allowed: false` и `physical_ground_truth: false`.

```bash
.venv/bin/python scripts/audit_additional_captures.py \
  --selection docs/other-models-selection-2026-10-09.json \
  --profile config/st-wheel-recovery-calibration.json \
  --weights runs/accuracy_10cm/models/yolo26m.pt \
  --video-dir /home/user/Downloads \
  --output-dir runs/other_models_reproduced --device cpu
```

Запуск использует настоящий `station.capture(temporal=True, review_fallback=True)`, детектор, сегментацию и поиск колёс. Это проверка захвата выбранных обнаружений; она не измеряет полноту обнаружения всех машин. Нужны ранее подготовленные модели из [инструкции установки](WHEEL_CAPTURE_2026-10-08.md). CPU разрешён явно только в офлайн-процессе.

Локальная галерея: `runs/other_models_20261009/index.html`. Полные исправленные снимки 2592×1944, маски, записи станции и полный журнал результатов находятся в `runs/other_models_20261009/replay/`. Все десять JPEG побайтово сверены с полными снимками, сохранёнными станцией; исходные кадры камеры и вырезки не подменяют их. Фото остаются локальными. Проверено отсутствие пересечения ID с прежними 78+7 каталожными примерами; пересечение с техническим аудитом — 10/10. Производственная база и калибровка не менялись.

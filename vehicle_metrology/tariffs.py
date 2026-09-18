"""Appendix 1, section 6 amended by order 37/ПЯ of 2026-09-08.

Ranges are transcribed literally, without rounding or filling gaps.
"""
from decimal import Decimal

CATEGORIES = {
    'car':'Легковой автомобиль', 'car_trailer':'Прицеп к легковому автомобилю',
    'boat_trailer':'Лодочный прицеп', 'bus':'Пассажирский автобус',
    'motorcycle':'Мотороллер / мотоцикл', 'truck':'Грузовой автомобиль',
    'truck_trailer':'Грузовой автомобиль + легковой / лодочный прицеп',
    'truck_capacity':'Грузовой автомобиль по грузоподъёмности',
    'tractor':'Тягач без прицепа', 'road_train':'Тягач с прицепом / полуприцепом',
    'oversize':'Тягач с тралом: негабаритный / тяжеловесный груз',
    'lowbed':'Автосостав: тягач + трал',
}
# code, lower, upper, rubles, lower inclusive, upper inclusive, description
BANDS = {
    'car': [('1.1',None,'4.00',1390,True,True,'Особо малый, до 4,00 м'),
            ('1.2','4.01','4.60',1650,True,True,'Малый, 4,01–4,60 м'),
            ('1.3','4.61','5.00',1900,True,True,'Средний, 4,61–5,00 м'),
            ('1.4','5.00',None,2200,False,True,'Большой, более 5,00 м')],
    'car_trailer':[('2',None,None,1200,True,True,'Прицеп к легковому автомобилю')],
    'boat_trailer':[('3',None,None,2000,True,True,'Лодочный прицеп')],
    'bus':[('4.1',None,'5.00',2800,True,True,'Малый, до 5,00 м'),
           ('4.2','5.01','10.00',3600,True,True,'Средний, 5,01–10,00 м'),
           ('4.3','10.01',None,5000,False,True,'Большой, более 10,01 м')],
    'motorcycle':[('5',None,None,380,True,True,'Все марки')],
    'truck':[('6.1',None,'7.09',2700,True,True,'Малый, до 7,09 м'),
             ('6.2','7.10','8.09',5350,True,True,'Средний, 7,10–8,09 м'),
             ('6.3','8.10','10.09',6450,True,True,'Большой, 8,10–10,09 м'),
             ('6.4','10.10','11.90',8500,True,True,'Особо большой, 10,10–11,90 м')],
    'truck_trailer':[('6.5','11.90',None,9000,False,True,'Грузовой автомобиль + прицеп (легковой, лодочный), на сцепке свыше 11,9 м')],
    'truck_capacity':[('6.6','15.001','24.00',9200,True,True,'Грузоподъёмность от 15,001 до 24,00 т'),
                      ('6.7','24.001','30.00',10950,True,True,'Грузоподъёмность от 24,001 до 30,00 т'),
                      ('6.8','30.001',None,14250,True,True,'Грузоподъёмность от 30,001 т и выше')],
    'tractor':[('7.1','7.10','11.90',5650,True,True,'Без прицепа, 7,10–11,90 м')],
    'road_train':[('7.2',None,'12.00',9050,True,True,'С прицепом, до 12,00 м'),
                  ('7.3','12.10','15.00',10800,True,True,'С прицепом, 12,10–15,00 м'),
                  ('7.4','15.10',None,14000,True,True,'С прицепом, от 15,10 м')],
    'oversize':[('7.5',None,None,22600,True,True,'Трал с негабаритным / тяжеловесным грузом')],
    'lowbed':[('8.1',None,'12.00',10750,True,True,'Тягач + трал до 12,00 м'),
              ('8.2','12.10','14.00',11850,True,True,'Тягач + трал 12,10–14,00 м'),
              ('8.3','14.10','16.00',14300,True,True,'Тягач + трал 14,10–16,00 м')],
}


def quote(category, length_m, manual_rub=None, load_capacity_t=None):
    if category not in CATEGORIES:
        raise ValueError('Выберите категорию ТС')
    length = None if length_m is None else Decimal(str(length_m))
    if length is not None and (not length.is_finite() or length <= 0):
        raise ValueError('Длина должна быть положительным числом')
    capacity = None if load_capacity_t is None else Decimal(str(load_capacity_t))
    if capacity is not None and (not capacity.is_finite() or capacity <= 0):
        raise ValueError('Грузоподъёмность должна быть положительным числом')
    by_capacity = category == 'truck_capacity'
    if by_capacity:
        length = capacity
    if manual_rub is not None:
        if not isinstance(manual_rub,int) or not 0 <= manual_rub <= 10000000:
            raise ValueError('Укажите целый тариф в рублях')
        return dict(amount_rub=manual_rub,code='manual',mode='manual',warnings=['Тариф установлен оператором'],boundary_m=None)
    bands=BANDS[category]
    fixed=len(bands)==1 and bands[0][1] is None and bands[0][2] is None
    warnings=[]
    category_options=[]
    amount=code=None
    if length is None and not fixed:
        warnings.append('Укажите грузоподъёмность по документам, в тоннах (не массу груза).' if by_capacity else
                        'Для расчёта тарифа нужна длина. Уточните измерение.')
    else:
        for c,lo,hi,price,li,ui,_ in bands:
            lower=lo is None or (length>=Decimal(lo) if li else length>Decimal(lo))
            upper=hi is None or (length<=Decimal(hi) if ui else length<Decimal(hi))
            if lower and upper:
                amount,code=price,c
                break
        if amount is None:
            if category=='truck' and length>Decimal('11.90'):
                warnings.append('Категория «Грузовой автомобиль» по длине заканчивается на 11,90 м. Для тягача с полуприцепом выберите «Тягач с прицепом / полуприцепом». Для других грузовиков уточните тип состава и грузоподъёмность по документам; длина не определяет категорию.')
                category_options=['road_train','truck_capacity','truck_trailer','lowbed','oversize']
            else:
                warnings.append(('Грузоподъёмность' if by_capacity else 'Длина')+' не попадает в однозначный диапазон Приложения №1. Нужен ручной тариф с причиной.')
    boundary=None
    if length is not None:
        bounds={Decimal(v) for b in bands for v in b[1:3] if v is not None}
        if bounds:
            nearest=min(bounds,key=lambda n:abs(length-n))
            if abs(length-nearest)<=Decimal('0.10'):
                boundary=float(nearest)
                warnings.append('Грузоподъёмность в пределах 0,10 т от границы тарифа. Проверьте документы.' if by_capacity else
                                'Длина в пределах 0,10 м от границы тарифа. Проверьте измерение.')
    if category in {'tractor','road_train','lowbed','oversize'}:
        warnings.append('Проверьте тип состава и полную длину с прицепом. Пункт 7.2 также содержит особое условие для ГАЗ / малотоннажных ТС от 12 м; для него используйте ручной тариф с причиной.')
    return dict(amount_rub=amount,code=code,mode='auto',warnings=warnings,
                boundary_m=None if by_capacity else boundary, boundary_t=boundary if by_capacity else None,
                category_options=category_options)


def catalog():
    return [dict(category=k,category_label=CATEGORIES[k],code=b[0],description=b[6],amount_rub=b[3])
            for k,rows in BANDS.items() for b in rows]

import xml.etree.ElementTree as ET
import numpy as np

def wczytaj_konfiguracje_xml(sciezka_do_pliku):

    tree = ET.parse(sciezka_do_pliku)
    root = tree.getroot()

    adam_defaults = {}
    adam_node = root.find('adam_defaults')
    if adam_node is not None:
        adam_defaults['alpha'] = float(adam_node.find('alpha').text)
        adam_defaults['beta1'] = float(adam_node.find('beta1').text)
        adam_defaults['beta2'] = float(adam_node.find('beta2').text)
        adam_defaults['eps'] = float(adam_node.find('eps').text)
        adam_defaults['max_iter'] = int(adam_node.find('max_iter').text)

    grupy_eksperymentow = {}
    
    for exp_node in root.find('experiments').findall('experiment'):
        exp_id = exp_node.attrib.get('id', 'Brak_ID')
        
        run_data = {
            'name': exp_node.find('name').text,
            'function_name': exp_node.find('function_name').text,
        }

        coords = [float(c.text) for c in exp_node.find('start_point').findall('coordinate')]
        run_data['start_point'] = np.array(coords)

        is_constrained_str = exp_node.find('is_constrained').text.strip().lower()
        run_data['is_constrained'] = (is_constrained_str == 'true')

        run_data['adam_overrides'] = {}
        hyper_node = exp_node.find('hyperparameters')
        if hyper_node is not None:
            for param in hyper_node.findall('param'):
                p_name = param.attrib['name']
                p_val = param.text
                if p_name == 'max_iter':
                    run_data['adam_overrides'][p_name] = int(p_val)
                else:
                    run_data['adam_overrides'][p_name] = float(p_val)

        if run_data['is_constrained']:
            penalty_node = exp_node.find('penalty_config')
            if penalty_node is not None:
                run_data['r'] = float(penalty_node.find('r').text)
                lista_ograniczen = []
                for const in penalty_node.find('constraints').findall('constraint'):
                    const_data = {
                        'type': const.attrib.get('type'),
                        'class_name': const.find('class_name').text,
                        'parameters': {}
                    }
                    for param in const.find('parameters').findall('param'):
                        const_data['parameters'][param.attrib['name']] = float(param.text)
                    lista_ograniczen.append(const_data)
                run_data['constraints'] = lista_ograniczen
            else:
                run_data['r'] = 1.0
                run_data['constraints'] = []
        else:
            run_data['r'] = 1.0
            run_data['constraints'] = []

        if exp_id not in grupy_eksperymentow:
            grupy_eksperymentow[exp_id] = {
                'id': exp_id,
                'name': run_data['name'],
                'runs': []
            }
        grupy_eksperymentow[exp_id]['runs'].append(run_data)

    return {
        'adam_defaults': adam_defaults,
        'experiment_groups': list(grupy_eksperymentow.values())
    }

import os

datasets = {'OfficeCaltech':
            {'root': '/shared/tale2/Shared/data/office/office_caltech_10',
             'domains': ['amazon', 'caltech', 'dslr', 'webcam'],
             'n_classes': 10,
             },
            'PACS':
                {'root': '/shared/tale2/Shared/data/PACS/kfold',
                 'domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
                 'n_classes': 7,
                 },
            'VLCS':
                {'root': '/shared/tale2/Shared/data/VLCS',
                 'domains': ['CALTECH', 'LABELME', 'SUN', 'PASCAL'],
                 'n_classes': 5,
                 },
            'OfficeHome':
                {'root': '/shared/tale2/Shared/data/OfficeHome',
                 'domains': ['Art', 'Clipart', 'Product', 'Real_world'],
                 'n_classes': 65,
                 }
            }


def mk_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)


data_list = ['OfficeCaltech', 'PACS', 'VLCS', 'OfficeHome']
methods = ['DANN', 'CDAN']

cfg_dir = './configs'
mk_dir(cfg_dir)

for data_name in data_list:
    data_set = datasets[data_name]
    data_cfg_dir = os.path.join(cfg_dir, data_name)
    mk_dir(data_cfg_dir)
    for method in methods:
        method_cfg_dir = os.path.join(data_cfg_dir, method)
        mk_dir(method_cfg_dir)

        for domain in data_set['domains']:
            cfg_fname = '%s_%s_%s.yaml' % (data_name, method, domain)
            cfg_file = open(os.path.join(method_cfg_dir, cfg_fname), "w")
            cfg_file.write("DAN:\n")
            cfg_file.write("  METHOD: %s\n" % method)
            cfg_file.write("DATASET:\n")
            cfg_file.write("  Name: %s\n" % data_name)
            cfg_file.write("  NUM_CLASSES: %s\n" % str(data_set['n_classes']))
            cfg_file.write("  ROOT: %s\n" % data_set['root'])
            cfg_file.write("  SOURCE: %s\n" % str([d for d in data_set['domains'] if d != domain]))
            cfg_file.write("  TARGET: %s\n" % str([domain]))
            cfg_file.write("  NUM_REPEAT: 5\n")
            cfg_file.write("SOLVER:\n")
            cfg_file.write("  TRAIN_BATCH_SIZE: 90\n")
            cfg_file.write("  TEST_BATCH_SIZE: 90\n")
            cfg_file.close()

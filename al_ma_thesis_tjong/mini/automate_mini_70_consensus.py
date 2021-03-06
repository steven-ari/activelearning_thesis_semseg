import os
from os.path import dirname as dr, abspath
from al_ma_thesis_tjong.mini import bulk_70_mini_getter as getter_mini


if __name__ == '__main__':

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    dir_name_input = 'bulk_70_mini_getter_from_consensus_90_001_'
    index_path_name = 'consensus_90_5_70_001'

    getter_mini.main(manual_seed_input=5, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    dir_name_input = 'bulk_70_mini_getter_from_consensus_90_002_'
    index_path_name = 'consensus_90_5_70_002'

    getter_mini.main(manual_seed_input=6, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    dir_name_input = 'bulk_70_mini_getter_from_consensus_90_003_'
    index_path_name = 'consensus_90_5_70_003'

    getter_mini.main(manual_seed_input=7, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    dir_name_input = 'bulk_70_mini_getter_from_consensus_90_004_'
    index_path_name = 'consensus_90_5_70_004'

    getter_mini.main(manual_seed_input=8, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    dir_name_input = 'bulk_70_mini_getter_from_consensus_90_005_'
    index_path_name = 'consensus_90_5_70_005'

    getter_mini.main(manual_seed_input=9, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)
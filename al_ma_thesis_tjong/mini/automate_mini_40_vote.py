import os
from os.path import dirname as dr, abspath
from al_ma_thesis_tjong.mini import bulk_40_mini_getter as getter_mini


if __name__ == '__main__':

    """source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    dir_name_input = 'bulk_40_mini_getter_from_vote_90_001_'
    index_path_name = 'vote_90_5_001'

    getter_mini.main(manual_seed_input=5, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    dir_name_input = 'bulk_40_mini_getter_from_vote_90_002_'
    index_path_name = 'vote_90_5_002'

    getter_mini.main(manual_seed_input=10, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    dir_name_input = 'bulk_40_mini_getter_from_vote_90_003_'
    index_path_name = 'vote_90_5_003'

    getter_mini.main(manual_seed_input=11, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)"""

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    dir_name_input = 'bulk_40_mini_getter_from_vote_90_004_'
    index_path_name = 'vote_90_5_004'

    getter_mini.main(manual_seed_input=10, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)

    source_path_input = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    dir_name_input = 'bulk_40_mini_getter_from_vote_90_005_'
    index_path_name = 'vote_90_5_005'

    getter_mini.main(manual_seed_input=12, source_path_input=source_path_input,
                     dir_name_input=dir_name_input, index_path_name=index_path_name)
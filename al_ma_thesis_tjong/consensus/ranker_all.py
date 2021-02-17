
from al_ma_thesis_tjong.consensus import ve_ranker as ve_ranker

from al_ma_thesis_tjong.consensus import consensus_ranker as ce_ranker

from al_ma_thesis_tjong.process.timer import Timer


if __name__ == '__main__':
    ve_ranker.main(0.2)
    ve_ranker.main(0.9)
    ce_ranker.main(0.2)
    ce_ranker.main(0.9)

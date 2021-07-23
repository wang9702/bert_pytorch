# -*- coding: utf-8 -*-
# @Time    : 2021/7/22 22:18
# @Author  : WangCheng
# @File    : xxxx.py


if __name__ == '__main__':
    with open('./data/corpus.txt', 'w', encoding='utf8') as f:
        sents = ['入院20天前患者无明显诱因出现左肩疼痛 \t 不伴颈部疼痛及右下肢疼痛麻木。', '入院前3天患者无明显诱因出现阵发性眩晕 \t 自诉两上肢乏力', \
                 '缘于入院前1余年前于我院诊为胃癌 \t 于2012-08-07在全麻上行根治术全胃切除术。', '入院前10+年患者偶然查出血压升高，190/120mmHg，伴头昏、头痛 \t 服用降压药物后好转。']
        for sent in sents:
            f.write(sent)
            f.write('\n')
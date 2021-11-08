import numpy as np
import re
from bs4 import BeautifulSoup

def remove_html(htmlstr):
    re_cdata = re.compile('//<![CDATA[[^>]*//]]>',re.I)#匹配CDATA
    re_script = re.compile('<s*script[^>]*>[^<]*<s*/s*scripts*>',re.I)#Script
    re_style=re.compile('<s*style[^>]*>[^<]*<s*/s*styles*>',re.I)#style
    re_h=re.compile('</?w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    re_br2 = re.compile("<[^>]+?>")
    re_br3 = re.compile("\[[^>]+?\]")

    s = htmlstr
    # s = re_cdata.sub('',htmlstr)#去掉CDATA
    # s = re_script.sub('',s) #去掉SCRIPT
    # s = re_style.sub('',s)#去掉style 
    s = re_br2.sub('',s)#将br转换为换行
    s = re_br3.sub('',s)#将br转换为换行

    s = re_h.sub('',s) #去掉HTML 标签
    s = re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    # blank_line=re.compile('n+')
    # s = blank_line.sub(' ',s)
    return s
def remove_time(str_sentence):
    str_sentence = re.sub('\(.*:.*\) ', "", str_sentence)
    # str_sentence = re.sub(r'\(.*', "", str_sentence)
    return str_sentence

def remove_newline(str_sentence):
    str_sentence = re.sub('\s+'," ",str_sentence)
    return str_sentence

def process_string(str_sentence):
    # print(str_sentence)
    str_sentence = remove_html(str_sentence)
    # str_sentence = BeautifulSoup(str_sentence, "html.parser").get_text()
    # print(str_sentence)
    # print()
    str_sentence = remove_time(str_sentence)
    str_sentence = remove_newline(str_sentence)
    return str_sentence

def split_file(data_dir = '/data/bairu/mt_dataset/', filename = 'opus.ha-en.tsv'):
    source_data = []
    target_data = []
    filtered = 0
    count = 0
    with open(data_dir + filename, 'r', encoding = 'utf-8') as f:
        for line in f:
            count += 1
            # if count >= 10000:
                # break
            try:
                source, target, _ = line.strip().split('\t', 2)
                source = source.lower()
                target = target.lower()
                source = process_string(source)
                target = process_string(target)
                if len(source) != 0 and len(target) != 0:
                    source_data.append(source)
                    target_data.append(target)
                else:
                    filtered += 1
                    print(line)
            except:
                # print(line)
                pass
    print("total data: ", len(source_data))
    print("filter num: ", filtered)
    return source_data, target_data
    

def write_file(data_dir = '/data/bairu/mt_dataset/opus/', filename = 'en.txt', data_list= []):
    with open(data_dir + filename, 'w', encoding = 'utf-8') as f:
        for item in data_list:
            f.write(item + '\n')

def read_file(data_dir = '/data/bairu/mt_dataset/dev/', filename = 'dev_en-ha.txt.en'):
    data_list = []
    filtered = 0
    with open(data_dir + filename, 'r', encoding = 'utf-8') as f:
        for line in f:
            sentence = line.strip().lower()
            sentence = process_string(sentence)
            if len(sentence) == 0:
                filtered += 1
                print(line)
                continue
            data_list.append(sentence)
    print("total data: ", len(data_list))
    print("filter num: ", filtered)
    return data_list

# source_corpus = []
# target_corpus = []
if __name__ == '__main__':
    data_dir = '/data/bairu/mt_dataset/'
    source_corpus, target_corpus = split_file()
    all_corpus = source_corpus + target_corpus
    write_file(data_list = source_corpus, filename = 'en.txt')
    write_file(data_list = target_corpus, filename = 'ha.txt')
    write_file(data_list = all_corpus, filename = 'all.txt')


    dev_en = read_file(filename = 'dev.en')
    dev_ha = read_file(filename = 'dev.ha')
    write_file(data_list = dev_en, filename = 'dev_en-ha.txt.en')
    write_file(data_list = dev_ha, filename = 'dev_en-ha.txt.ha')



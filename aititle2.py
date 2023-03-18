# WEBスクレイピングに必要なライブラリをインストール


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import openpyxl


from flask import Flask,render_template, request  # request追加
from wtforms import Form, StringField, validators, SubmitField

from webdriver_manager.chrome import ChromeDriverManager
#from selenium.webdriver.chrome.service import Service

#driver_path = "./chromedriver"


print("インポート完了")


# タイトル作成部分を追加する

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW,get_linear_schedule_with_warmup

# 学習済みモデルをHugging Face model hubからダウンロードする
model_dir_name = "sonoisa/t5-qiita-title-generation"

# トークナイザー（SentencePiece）

tokenizer = T5Tokenizer.from_pretrained(model_dir_name, is_fast=True)


# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_name)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()

# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変
# from __future__ import unicode_literals  #python2/3両方使う場合
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                    '\u3040-\u309F',  # HIRAGANA
                    '\u30A0-\u30FF',  # KATAKANA
                    '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                    '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                    ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
            '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

import re

CODE_PATTERN = re.compile(r"```.*?```", re.MULTILINE | re.DOTALL)
LINK_PATTERN = re.compile(r"!?\[([^\]\)]+)\]\([^\)]+\)")
IMG_PATTERN = re.compile(r"<img[^>]*>")
URL_PATTERN = re.compile(r"(http|ftp)s?://[^\s]+")
NEWLINES_PATTERN = re.compile(r"(\s*\n\s*)+")

def clean_markdown(markdown_text):
    markdown_text = CODE_PATTERN.sub(r"", markdown_text)
    markdown_text = LINK_PATTERN.sub(r"\1", markdown_text)
    markdown_text = IMG_PATTERN.sub(r"", markdown_text)
    markdown_text = URL_PATTERN.sub(r"", markdown_text)
    markdown_text = NEWLINES_PATTERN.sub(r"\n", markdown_text)
    markdown_text = markdown_text.replace("`", "")
    return markdown_text

def normalize_text(markdown_text):
    markdown_text = clean_markdown(markdown_text)
    markdown_text = markdown_text.replace("\t", " ")
    markdown_text = normalize_neologd(markdown_text).lower()
    markdown_text = markdown_text.replace("\n", " ")
    return markdown_text

def preprocess_material(markdown_text):
    return "body: " + normalize_text(markdown_text)[:4000]

def postprocess_title(title):
    return re.sub(r"^title: ", "", title)

    
#WEBスクレイピングの部分を関数化

def webscr(keywords):
    INTERVAL = 2.0
    
    options = webdriver.ChromeOptions()

    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions') 
    
    #service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(ChromeDriverManager().install(),executable_path='./chromedriver',options=options)

    driver.get('https://www.google.com/')       # Googleを開く

    search = driver.find_element_by_name("q")   # HTML内で検索ボックス(name='q')を指定する
    # driver.find_element(By.CLASS_NAME, 'q')
    
    search.send_keys(keywords)    # 検索ワードを送信する
    
    #Google検索ボタンをクリック
    search.submit()                         # 検索を実行
    time.sleep(INTERVAL)
    print("2秒待ちな") 
    
    #検索結果の一覧を取得する
    titles = []
    links = []
    
    result = {
    'タイトル': titles,
    'URL': links
    }
    
    flag = False
    while True:
        elems_h3 = driver.find_elements_by_xpath('//a/h3')
        for elem_h3 in elems_h3:
            link = elem_h3.find_element_by_xpath('..').get_attribute('href')
            title = elem_h3.text
            
            if not elem_h3.text == '':
                titles.append(title)
                links.append(link)
        
                
            if len(titles) >= 10: #抽出する件数を指定
                flag = True
                break
        if flag:
            break
        driver.find_element_by_id('pnnext').click()
        time.sleep(INTERVAL)
    driver.quit() 
    
    return titles
    return links
    return result
    

def make_material(titles):
    material = '.'.join(titles)  # スクレイピングしてきたタイトルを改行して1つの文章にする
    
    print(material)
    return material



def create_title(material):


    MAX_SOURCE_LENGTH = 512  # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = 64   # 生成されるタイトルの最大トークン数

    # 推論モード設定
    trained_model.eval()

    # 前処理とトークナイズを行う
    inputs = [preprocess_material(material)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']

    input_mask = batch['attention_mask']
    
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    # 生成処理を行う
    output = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask, 
        max_length=MAX_TARGET_LENGTH,
        return_dict_in_generate=True, output_scores=True,
        temperature=2.0,            # 生成にランダム性を入れる温度パラメータ
        num_beams=10,               # ビームサーチの探索幅
        diversity_penalty=0.5,      # 生成結果の多様性を生み出すためのペナルティ
        num_beam_groups=10,         # ビームサーチのグループ数
        num_return_sequences=5,     # 生成する文の数
        repetition_penalty=1.0      # 同じ文の繰り返し（モード崩壊）へのペナルティ
    )    


    # 生成されたトークン列を文字列に変換する
    generated_title = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in output.sequences]

    ai_title = []

    # 生成されたタイトルを表示する
    for i, title in enumerate(generated_title):
        print(f"{i+1:2}. {postprocess_title(title)}")
        ai_title.append({postprocess_title(title)})
    return ai_title



print("下準備OK")

# Flask のインスタンスを作成
app = Flask(__name__)



#入力されたキーワードを処理する

# WTForms を使い、index.html 側で表示させるフォームを構築します。
class InputForm(Form):
    InputFormTest = StringField('キーワードを入力してENTER',
                    [validators.InputRequired()])
    # HTML 側で表示する submit ボタンの表示
    submit = SubmitField('タイトル作成')

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])

def input():
    # WTForms で構築したフォームをインスタンス化
    form = InputForm(request.form)

    # POST メソッドの条件の定義
    if request.method == 'POST':

        # 入力のvaridateで異常の場合
        if form.validate() == False:
            return render_template('index.html', outputname='再チャレンジ')
        # 正常に動いた場合
        else:
            search_key = request.form['InputFormTest']
            title_sum = webscr(keywords=search_key)
            mate = make_material(titles=title_sum)
            ai_title = create_title(material=mate)
            return render_template('result.html', outputname=ai_title)

    # GET メソッドの定義　初回のページ
    elif request.method == 'GET':
        return render_template('index.html', forms=form)





# データフレームを作成する
# df = pd.DataFrame(result)
# print(df)
# csvで出力する

# df.to_csv(f'google_search_{keywords}.csv', encoding="shift-jis")

# 一度ファイルオブジェクトをエラー無視して、書き込みで開くようにする
# with open(f"google_search_{keywords}.csv", mode="w", encoding="shift-jis", errors="ignore", newline="") as f:
#     df.to_csv(f,index=False)  # ここでデータフレームを開いたファイルにcsvで書き込む









# アプリケーションの実行
if __name__ == '__main__':
    app.run()
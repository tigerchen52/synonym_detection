import pprint

import pymysql

from synonym_detect import data_utils, word2vec_model, semantic_network_model, Levenshtein_model


def run(corpus_path, input_word_path, stop_word_path='../input/stop_words.txt', top_k=10):

    top_k += 1

    word2id, word_list, id2word, input_word_code_dict, input_word_ids = data_utils.preprocess_file(corpus_path, input_word_path, stop_word_path)

    # use_w2v_model
    rs1 = word2vec_model.synonym_detect(input_word_code_dict, top_k)
    pprint.pprint(rs1)

    # use_sn_model
    rs2 = semantic_network_model.synonym_detect(
        corpus_path=corpus_path,
        input_word_ids=input_word_ids,
        input_word_code_dict=input_word_code_dict,
        id2word=id2word,
        word2id=word2id,
        top_k=top_k,
        win_len=5,
        process_number=1
    )
    pprint.pprint(rs2)

    # use_leven_model
    l_model = Levenshtein_model.Levenshtein_model(
        input_word=list(input_word_code_dict.keys()),
        candidate_word=word_list,
        process_number=1,
        if_use_pinyin=False,
        pinyin_weight=0,
        top_k=top_k
    )
    rs3 = l_model.multipro_synonym_detect(input_word_code_dict)
    pprint.pprint(rs3)


    conn = pymysql.connect(
        host='tslow.cn',
        user='test',
        password='test',
        database='testdb',
        charset='utf8mb4',
    )

    cursor = conn.cursor()

    def exec_sql(sql, cursor, first=False):
        print(sql)
        cursor.execute(sql)
        if first:
            r = cursor.fetchone()
            if r and len(r) == 1:
                return r[0]
            else:
                return r
        else:
            return cursor.fetchall()


    # ========== init =============
    sql = """
    CREATE TABLE IF NOT EXISTS words (
        id INT auto_increment PRIMARY KEY ,
        word CHAR(10) NOT NULL UNIQUE
    )ENGINE=innodb DEFAULT CHARSET=utf8mb4;
    """
    exec_sql(sql, cursor)


    sql = """
    CREATE TABLE IF NOT EXISTS synonym (
        id INT auto_increment PRIMARY KEY ,
        word1_id INT,
        word2_id INT,
        w2v_score FLOAT,
        sn_score FLOAT,
        leven_score FLOAT
    )ENGINE=innodb DEFAULT CHARSET=utf8mb4;
    """
    exec_sql(sql, cursor)


    # ========== make words =============
    for word in input_word_code_dict.keys():
        sql = f"select count(*) from words where word='{word}';"
        count = exec_sql(sql, cursor, True)
        if not count:
            sql = f"insert into words(word) values ('{word}');"
            exec_sql(sql, cursor)

    for synonyms in list(rs1.values()) + list(rs2.values()) + list(rs3.values()):
        for nword, score in synonyms:
            sql = f"select count(*) from words where word='{nword}';"
            count = exec_sql(sql, cursor, True)
            if not count:
                sql = f"insert into words(word) values ('{nword}');"
                exec_sql(sql, cursor)

    conn.commit()

    words_map = {}
    sql = f"select * from words;"
    res = exec_sql(sql, cursor)
    print(res)
    for line in res:
        word_id, word = line
        words_map[word] = word_id

    print(words_map)


    # ========== make synonym =============
    for word, synonyms in rs1.items():
        word1_id = words_map[word]
        for nword, score in synonyms:
            word2_id = words_map[nword]
            sql = f"select id from synonym where word1_id={word1_id} and word2_id={word2_id};"
            sid = exec_sql(sql, cursor, True)
            print(word, nword, sid)
            if sid:
                sql = f"update synonym set w2v_score={score} where id={sid};"
                exec_sql(sql, cursor)
            else:
                sql = f"insert into synonym (word1_id, word2_id, w2v_score) values ({word1_id}, {word2_id}, {score});"
                exec_sql(sql, cursor)

    for word, synonyms in rs2.items():
        word1_id = words_map[word]
        for nword, score in synonyms:
            word2_id = words_map[nword]
            sql = f"select id from synonym where word1_id={word1_id} and word2_id={word2_id};"
            sid = exec_sql(sql, cursor, True)
            print(word, nword, sid)
            if sid:
                sql = f"update synonym set sn_score={score} where id={sid};"
                exec_sql(sql, cursor)
            else:
                sql = f"insert into synonym (word1_id, word2_id, sn_score) values ({word1_id}, {word2_id}, {score});"
                exec_sql(sql, cursor)

    for word, synonyms in rs3.items():
        word1_id = words_map[word]
        for nword, score in synonyms:
            word2_id = words_map[nword]
            sql = f"select id from synonym where word1_id={word1_id} and word2_id={word2_id};"
            sid = exec_sql(sql, cursor, True)
            if sid:
                sql = f"update synonym set leven_score={score} where id={sid};"
                exec_sql(sql, cursor)
            else:
                sql = f"insert into synonym (word1_id, word2_id, leven_score) values ({word1_id}, {word2_id}, {score});"
                exec_sql(sql, cursor)


    conn.commit()
    cursor.close()
    conn.close()


corpus_path = '../input/三体.txt'
input_word_path = '../input/input_words.txt'
stop_word_path = '../input/stop_words.txt'
top_k = 10

if __name__ == '__main__':
    run(corpus_path, input_word_path, stop_word_path, top_k)

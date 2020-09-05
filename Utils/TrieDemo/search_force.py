#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys, os

if __name__ == "__main__":
    dic_words = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    tag_word = sys.argv[4]

    f_out = open(output_file, 'w')
    with open(dic_words, 'r') as f1:
        lines = f1.readlines()
        for pword in lines:
            pword = pword.strip()
            with open(input_file, 'r') as f2:
                lines = f2.readlines()
                for cid in lines:
                    cid = cid.strip()
                    if pword in cid:
                        res_line = pword + "\t" + cid + "\t" + tag_word + "\n"
                        f_out.write(res_line)
                        f_out.flush()

    print "write done"

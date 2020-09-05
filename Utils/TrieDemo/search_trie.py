#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys, os

class TrieTree(object):
    def __init__(self):
        self.tree = {}

    def add(self, word):
        tree = self.tree
        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                tree[char] = {}
                tree = tree[char]
        tree['exist'] = True

    def addFromFile(self, wordsFile):
        with open(wordsFile, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                self.add(line.strip())

    def search(self, word):
        tree = self.tree
        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                return False
        if "exist" in tree and tree["exist"] == True:
            return True
        else:
            return False

    def searchInText(self, text):
        """从text中搜索属于tree词典的字符串，返回结果列表"""
        wordsFound = []
        for i in range(len(text)):
            for j in range(len(text), i, -1):
                word = text[i: j]
                if self.search(word):
                    wordsFound.append(word)
        return wordsFound
			

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "input error"
        exit(1)
    words_file = sys.argv[1]
    if not os.path.exists(words_file):
        print "The words file doesn't exist"
    input_file = sys.argv[2]
    if not os.path.exists(input_file):
        print "The input file doesn't exist"
    output_file = sys.argv[3]
    if os.path.exists(output_file):
        os.remove(output_file)
    tag_word = sys.argv[4]

    # Init the trie tree
    tree = TrieTree()
	tree.addFromFile(words_file)

    # use the trie tree 
    f_out = open(output_file, 'w')
    with open(input_file, 'r') as fr:
        for line in fr.readlines():
            found_words = tree.searchFromText(line.strip())
            if len(found_words) > 0:
				res_line =""
                for word in found_words:
                    res_line += line + "\t" + word + "\t" + tag_word + "\n"
                f_out.write(res_line)
                f_out.flush()

	
# usage: ./search_trie.py ~/users/liuyao58/attr/result/pword.txt ~/users/liuyao58/resource/cid12_name.txt ./pword_tag.txt 1
# reference: ~/users/liuyao58/code/shop_in_pipline_all/py/trie.py


tree = TrieTree()
tree.add("iphone")
tree.add("小米")
print(tree.search("iphone"))
print(tree.search("小米"))
print(tree.search("iphone7"))
print(tree.search("小米6"))
print(tree.search("iphoneplus"))
print(tree.search("小米c"))

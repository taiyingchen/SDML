# Regular expressions
import re
# Use Inflect for singular-izing words
import inflect
# Gensim for learning phrases and word2vec
import gensim

# For some reason, inflect thinks that there is a singular form of 'mass', namely 'mas' 
# and similarly for gas. Please add any other exceptions to this list! 
p = inflect.engine()
p.defnoun('mass', 'mass|masses')
p.defnoun('gas', 'gas|gases')
p.defnoun('gas', 'gas|gasses')     # Other spelling
p.defnoun('gaas', 'gaas')          #GaAs ;)
p.defnoun('gapless', 'gapless')
p.defnoun('haas', 'haas')

# Check if a string has digits
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
            
# Return the singular form of a word, if it exists
def singularize(word):
    try:
        # p.singular_word() returns the singular form, but 
        # returns False if there is no singular form (or already singular)
        
        # So, if the word is already singular, just return the word
        if not p.singular_noun(word):
            return word
        else:
            # And otherwise return the singular version
            return p.singular_noun(word)
       
    except Exception as e:
        print("Euh? What's this? %s"%word)
        print("This caused an exception: ", e)
        return word
    
def stripchars(w, chars):
    return "".join( [c for c in w if c not in chars] ).strip('\n')
   
# Parse a title into words
def parse_title(title):
    """
    # Extract the year
    year, rest = title.split(' ', 1)
    year = int(year[0:])
    # Then the month
    month, title = rest.split(' ', 1)
    month = int(month[0:])
    """
    # Then, for every word in the title:
    # 1) Split the title into words, by splitting it on spaces ' ' and on '-' (de-hyphenate words).
    # 2) Turn each of those resulting words into lowercase letters only
    # 3) Strip out any weird symbols (we don't want parenthesized words, not ; at the end of a word, etc)
    # 4) Also, we don't want to have digits.. my apologies to all the material studies on interesting compounds!
    title = gensim.parsing.preprocessing.remove_stopwords(title) # Remove stopwords
    words = re.split( ' |-|\\|/', title.lower())
    wordlist = []
    for i in range(len(words)):
        w = words[i]
        
        # Skip if there is no word, or if we have numbers
        if len(w) < 1 or hasNumbers(w):
            continue

        # If it is (probably) math, let's skip it
        if w[0] == '$' and w[-1] == '$':
            continue
            
        # Remove other unwanted characters
        w = stripchars(w, '\\/$(){}.<>,;:_"|\'\n `?!#%')
        # Get singular form
        w = singularize(w)
        
        # Skip if nothing left, or just an empty space
        if len(w) < 2 or w == ' ':
            continue
            
        # Append to the list
        wordlist.append(w)
        
    # return  year, month, wordlist
    return wordlist

    # Previous versions
    #return year, month [singularize(stripchars(w, ['\\/$(){}.<>,;:"|\'\n '])) for w in re.split(' |-|\\|/',title.lower()) if not hasNumbers(w)]
    #return year, month, [singularize(w.strip("\\/$|[](){}\n;:\"\',")) for w in re.split(' |-',title.lower()) if not hasNumbers(w)]

def get_titles_for_years(all_titles, years):
    """ Return list of all titles for given years (must be a list, even if only one)"""
    collectedtitles = []
    for k in years:
        allmonthtitles = []
        for m in all_titles[k].keys():
            allmonthtitles = allmonthtitles + all_titles[k][m]
            
        collectedtitles = collectedtitles + allmonthtitles
    return collectedtitles

def get_ngrams(sentences):
    """ Detects n-grams with n up to 4, and replaces those in the titles. """
    # Train a 2-word (bigram) phrase-detector
    bigram_phrases = gensim.models.phrases.Phrases(sentences)
    
    # And construct a phraser from that (an object that will take a sentence
    # and replace in it the bigrams that it knows by single objects)
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    
    # Repeat that for trigrams; the input now are the bigrammed-titles
    ngram_phrases = gensim.models.phrases.Phrases(bigram[sentences])
    ngram         = gensim.models.phrases.Phraser(ngram_phrases)
    ngram.save('ngram.phraser')
    
    # !! If you want to have more than 4-grams, just repeat the structure of the
    #    above two lines. That is, train another Phrases on the ngram_phrases[titles],
    #    that will get you up to 8-grams. 
    
    # Now that we have phrasers for bi- and trigrams, let's analyze them
    # The phrases.export_phrases(x) function returns pairs of phrases and their
    # certainty scores from x.
    bigram_info = {}
    for b, score in bigram_phrases.export_phrases(sentences):
        bigram_info[b] = [score, bigram_info.get(b,[0,0])[1] + 1]
        
    ngram_info = {}
    for b, score in ngram_phrases.export_phrases(bigram[sentences]):
        ngram_info[b] = [score, ngram_info.get(b,[0,0])[1] + 1]
            
    # Return a list of 'n-grammed' titles, and the bigram and trigram info
    return [ngram[t] for t in sentences], bigram_info, ngram_info

# !!! THIS SECTION HAS NOT YET BEEN UPDATED
# !!! IT WILL WORK, BUT IT TAKES A *VERY* LONG 
# !!! TIME. HAS TO SWITCH TO LIST COMPREHENSION

# Parse abstract into sentences 
def parse_abstract(file):
    # Buffer for storing the file
    abstr = open(file, "r").read()
            
    sentences = []

    # Clean up abstract
    abstr.lower()   
    abstr.replace('\'', '')
    abstr.replace('\"', '')

    # Extract sentences and split into words
    end = abstr.find('.') 
    while end != -1:
        sentence = abstr[:end].replace('\n', ' ')
       
        # Sanitize the words
        words = re.split( ' |-|\\|/', sentence.lower() )
        wordlist = []
        for i in range(len(words)):
            w = words[i]

            # Skip if there is no word, or if we have numbers
            if len(w) < 1 or hasNumbers(w):
                continue

            # If it is (probably) math, let's skip it
            if w[0] == '$' and w[-1] == '$':
                continue

            # Remove other unwanted characters
            w = stripchars(w, '\\/$(){}.<>,;:_"|\'\n `?!#%')
            # Get singular form
            w = singularize(w)

            # Skip if nothing left, or just an empty space
            if len(w) < 1 or w == ' ':
                continue

            # Append to the list
            wordlist.append(w)
        
        sentences.append( wordlist )
        abstr = abstr[end+1:]
        end = abstr.find('.')

    return sentences
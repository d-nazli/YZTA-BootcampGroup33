# -*- coding: utf-8 -*-

df.rename(columns={'statement': 'original_statement'}, inplace=True)
df['statement']=df['original_statement'].str.lower()
df.head()

def remove_patterns(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove markdown-style links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove handles (that start with '@')
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation and other special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
df['statement'] = df['statement'].apply(remove_patterns)
df.head()

df['tokens'] = df['statement'].apply(word_tokenize)
df.head()

stemmer=PorterStemmer()
def stem_tokens(tokens):
    return ' '.join(stemmer.stem(str(token)) for token in tokens)

df['tokens_stemmed'] = df['tokens'].apply(stem_tokens)
df.head()

statuses = df['status'].unique()

# Her mental durum için kelime bulutu oluşturma
def color_func(word, font_size, position, orientation, random_state=101, **kwargs):
    return random.choice(colors)

for status in statuses:
    # Mevcut durum için belirteç verilerinin filtrelenmesi
    tokens_data = ' '.join(df[df['status'] == status]['tokens'].dropna().apply(lambda x: ' '.join(x)).tolist())

    #Kelime bulutunu çalıştırma
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func).generate(tokens_data)
    
    # Kelime bulutu görselleştirmesi
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    plt.title(f'WordCloud for Status: {status}')

    plt.show()

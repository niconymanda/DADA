import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

def find_unique_authors_create_dataFrame(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    unique_authors = set()
    authors_list = []
    ids_list = []
    quotes_list = []
    labels_list = []
    author_id_map = {}
    current_id = 0
    for item in data:
        author = item.get('speaker')
        if author:
            if author not in author_id_map:
                    author_id_map[author] = current_id
                    current_id += 1
            unique_authors.add(author)
            authors_list.append(author)
            ids_list.append(author_id_map[author])
            quotes_list.append(item.get('content'))
            labels_list.append(item.get('label'))
    data = pd.DataFrame({
        'AuthorID': ids_list,
        'Author': authors_list,
        'Quote': quotes_list,
        'Label': labels_list
    })
    return unique_authors, data, author_id_map

def add_quotes_from_brainy(data, folder_path, author_id_map, unique_authors):
    authors_list = []
    ids_list = []
    quotes_list = []

    all_files = os.listdir(folder_path)
    for file in all_files:
        if file.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file)
            with open(csv_file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    author = row.get('author')
                    author = author.split(',')[0]  
                    if 'Franklin D. Roosevelt' == author or 'John F. Kennedy' ==  author or 'Harry S Truman'==author or author in unique_authors:
                        author = author.replace('Franklin D. Roosevelt', 'FDR').replace('John F. Kennedy', 'JFK').replace('Harry S Truman', 'Harry Truman')
                        authors_list.append(author)
                        ids_list.append(author_id_map[author])
                        quotes_list.append(row.get('title') or row.get('quote'))


    # Create DataFrame
    data_csv = pd.DataFrame({
        'AuthorID': ids_list,
        'Author': authors_list,
        'Quote': quotes_list
    })

    data_csv['Label'] = 'bona-fide'
    
    data = pd.concat([data, data_csv], ignore_index=True)
    return data


def plot_author_quotes_count(data):
    quote_counts = data['Author'].value_counts()
    plt.figure(figsize=(10, 8))
    quote_counts.plot(kind='bar')
    plt.title('Number of Quotes per Author')
    plt.xlabel('Author')
    plt.ylabel('Number of Quotes')
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig('Data/author_quotes_count.png')
    return quote_counts


def main():
    json_file_path = 'Data/InTheWild/wild_transcription_meta.json'
    unique_authors, data, author_id_map = find_unique_authors_create_dataFrame(json_file_path)
    folder_path = 'Data/BrainyQuotes'
    data = add_quotes_from_brainy(data, folder_path, author_id_map, unique_authors)
    data.to_csv('Data/finalData.csv', index=False)
    quotes_count = plot_author_quotes_count(data)
   
    # Print table with authors and their quotes count
    print(f"{'Author ID':<10} {'Author Name':<30} {'Quotes Count':<15}")
    print("-" * 55)
    for author in author_id_map:
        author_id = author_id_map[author]
        print(f"{author_id:<10} {author:<30} {quotes_count[author]:<15}")
    return data

if __name__ == '__main__':
    data = main()
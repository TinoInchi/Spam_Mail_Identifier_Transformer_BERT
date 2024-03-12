import re

'''
    Important notice: This cleaning is based on the dataset I written and should be adjustet properly.
    Most of them are just to make sure, that no irrelevant data is feeded into the learning process, since all those
    are replaced with an empty string. Further all irrelevant symbols are replaced, since mails can contain, random symbols
    which does not effect the content.
    Further fill the spam labeler as you want, since you decide, what mails are spam for you, here are just some preparations.
'''

def take_only_adresses(sender):
    '''
    input: string with sender
    output: string with sender
    function: in Gmail, company mails are listed as 'name' <'adress'>
              To only take the adress, this function is made
    '''
    sender_array = sender.split()
    if len(sender_array) > 1:
        sender_email = sender_array.pop()
    else:
        sender_email = sender_array[0]
    sender_email_temp = ''
    if sender_email[0] == '<':
        for count in range(1,len(sender_email)-1,1):
            sender_email_temp +=  sender_email[count]
        sender_email = sender_email_temp
    return sender_email

def get_rid_of_symbols(content):
    '''
    input: string
    output: string
    function: takes out all symbols besides: .,;:?!
              as punctuation are important for BERT to learn  
    '''
    content = content.replace('\n','').replace('\r','')
    content_no_symbols = re.sub("[^a-zA-Z0-9.,;:?!]"," ",content)
    return content_no_symbols

def get_rid_of_all_symbols(content):
    '''
    input: string
    output: string
    function: takes out all symbols besides and all new lines, etc.
    '''
    content = content.replace('\n','').replace('\r','')
    content_no_symbols = re.sub("[^a-zA-Z0-9]"," ",content)
    return content_no_symbols

def delete_not_decoded_text(content):
    '''
    input: string
    output: string
    function: error handles not readable text due to failed decoding
    '''
    if content.startswith(' utf'):
        content = ''
    return content


def spam_labeler(data):
    '''
    input: pandas dataframe containing columns [sender,subject,text,label]
    function: labels all spam mails to 1
    '''

    spam_mail_adresses = ['Your spam mail adresses']
    for adress in spam_mail_adresses:
        data.loc[data['sender'] == adress,['label']] = 1

    data.loc[data['sender'].str.contains('your string containing something spamy'),['label']] = 1
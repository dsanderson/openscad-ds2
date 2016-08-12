import glob, mailbox, collections, tqdm

Email = collections.namedtuple('Email',['date','sender','message','quotes','subject','is_reply'])

def strip_quotes(content):
    lines = content.split('\n')
    c_lines = []
    q_lines = []
    for l in lines:
        if len(l.strip())>0 and l.strip()[0] == '>':
            q_lines.append(l)
        else:
            c_lines.append(l)
    content = '\n'.join(c_lines)
    quotes = '\n'.join(q_lines)
    return content, quotes

def get_messages(directory='./mailing_list/*.txt'):
    mailbox_files = glob.glob(directory)
    messages = []
    for f in tqdm.tqdm(mailbox_files):
        mb = mailbox.mbox(f)
        mb = list(mb)
        for m in mb:
            msg, quotes = strip_quotes(m.get_payload())

            email = Email(date = m.get('Date'), sender = m.get('From'), subject = m.get('Subject'), message=msg, quotes=quotes, is_reply=('In-Reply-To' in m.keys()))
            messages.append(email)

    #root_msgs = [m for m in messages if not m.is_reply]
    return messages

def get_root(emails):
    return [m for m in emails if not m.is_reply]

def strip_footer(emails):
    stripped = []
    for email in emails:
        msg = email.message
        msg = ' '.join(msg.split('--')[:-1])
        if len(msg)<1:
            msg = email.message
        em2 = Email(date = email.date, sender = email.sender, subject = email.subject, message = msg, quotes = email.quotes, is_reply=email.is_reply)
        stripped.append(em2)
    return stripped

def bundle_email(emails):
    '''return a dict of list of indecies, of emails that are in the same thread'''
    roots = get_root(emails)
    root_dict = {j:[] for j,r in enumerate(roots)}
    for i,e in enumerate(emails):
        if e.subject!=None:
            for j,r in enumerate(roots):
                if r.subject != None:
                    if r.subject in e.subject:
                        root_dict[j].append(i)
    return root_dict

if __name__ == '__main__':
    messages = get_messages()
    print len(messages), len(get_root(messages))

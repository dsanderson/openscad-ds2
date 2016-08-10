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

mailbox_files = glob.glob('./mailing_list/*.txt')

messages = []

for f in tqdm.tqdm(mailbox_files):
    mb = mailbox.mbox(f)
    mb = list(mb)
    for m in mb:
        msg, quotes = strip_quotes(m.get_payload())

        email = Email(date = m.get('Date'), sender = m.get('From'), subject = m.get('Subject'), message=msg, quotes=quotes, is_reply=('In-Reply-To' in m.keys()))
        messages.append(email)

root_msgs = [m for m in messages if not m.is_reply]

print len(messages), len(root_msgs)

from vowpal import *

DATA = [
    [0.4, {'body' : {'height' : 0.8, 'weight' : 0.3}, 'age' : {'age' : 0.4}, 'sports' : { 'football' : None }}],
    [0.7, {'body' : {'height' : 0.8, 'weight' : 0.3}, 'age' : {'age' : 0.3}, 'sports' : { 'soccer' : None }}],
    [0.2, {'body' : {'height' : 0.8, 'weight' : 0.5}, 'age' : {'age' : 0.7}, 'sports' : { 'soccer' : None }}],
    [0.7, {'body' : {'height' : 0.9, 'weight' : 0.3}, 'age' : {'age' : 0.2}, 'sports' : { 'track' : None }}],
    [0.9, {'body' : {'height' : 0.7, 'weight' : 0.3}, 'age' : {'age' : 0.3}, 'sports' : { 'track' : None }}],
    [0.6, {'body' : {'height' : 0.7, 'weight' : 0.7}, 'age' : {'age' : 0.2}, 'sports' : { 'track' : None }}],
    [None, {'body' : {'height' : 0.7, 'weight' : 0.2}, 'age' : {'age' : 0.3}, 'sports' : { 'track' : None }}],
    [None, {'body' : {'height' : 0.7, 'weight' : 0.2}, 'age' : {'age' : 0.3}, 'sports' : { 'soccer' : None }}],
]

def test_predict_from_examples():
    examples = []
    for i in xrange(len(DATA)):
        (value, all_sections) = DATA[i]
        ex = VowpalExample(i, value)
        for (namespace, section) in all_sections.items():
            ex.add_section(namespace, section)
        examples.append(ex)
    train = examples[:-2]
    test = examples[-2:]
    vw = Vowpal('/Users/shilad/Downloads/JohnLangford-vowpal_wabbit-9a1da62/vw', './vw.%s', {'--passes' : '10' })
    preds = vw.predict_from_examples(train, test)
    for (id, value) in preds:
        print 'prediction for %s is %s' % (id, value)

def test_predict_from_example_stream():
    stream = ExampleStream('vw.stream.txt')
    examples = []
    for i in xrange(len(DATA)):
        (value, all_sections) = DATA[i]
        ex = VowpalExample(i, value)
        for (namespace, section) in all_sections.items():
            ex.add_section(namespace, section)
        stream.add_example(ex)
    train = examples[:-2]
    test = examples[-2:]
    vw = Vowpal('/Users/shilad/Downloads/JohnLangford-vowpal_wabbit-9a1da62/vw', './vw.%s', {'--passes' : '10' })
    preds = vw.predict_from_example_stream(stream)
    for (id, value) in preds:
        print 'prediction for %s is %s' % (id, value)

def test_predict_from_file():
    f = open('vw.file.txt', 'w')
    examples = []
    for i in xrange(len(DATA)):
        (value, all_sections) = DATA[i]
        ex = VowpalExample(i, value)
        for (namespace, section) in all_sections.items():
            ex.add_section(namespace, section)
        f.write(str(ex) + '\n')
    f.close()
    vw = Vowpal('/Users/shilad/Downloads/JohnLangford-vowpal_wabbit-9a1da62/vw', './vw.%s', {'--passes' : '10' })
    preds = vw.predict_from_file('vw.file.txt')
    for (id, value) in preds:
        print 'prediction for %s is %s' % (id, value)

if __name__ == '__main__':
    #test_predict_from_examples()
    #test_predict_from_example_stream()
    test_predict_from_file()

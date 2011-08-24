import os
import string
import subprocess


class VowpalRecord:
   '''A single record that Vowpal predicts or learns.'''

   __slots__ = ('value', 'id', 'sections', 'SECTION_NAME_KEY')

   def __init__(self, id, value=None):
      self.SECTION_NAME_KEY = '__section_name__'
      self.value = value
      self.id = id
      self.sections = []    # list of dictionaries

   def addSection(self, name, section):
      '''
         Adds a new section of features for the record.
         Name is the namespace of the section.
         Section is a dictionary:
            Keys are feature names.
            Values are feature values or None for unary features.
         Namespaces are useful for creating interactions (see vowpal wiki).
      '''
      section[self.SECTION_NAME_KEY] = name
      self.sections.append(section)

   def __str__(self):
      sectionStrs = []
      if self.value in (None, ''):
         sectionStrs.append('%s %s' % (1.0, self.id))
      else:
         sectionStrs.append('%s %s %s' % (self.value, 1.0, self.id))
      for s in self.sections:
         tokens = [s[self.SECTION_NAME_KEY]]
         for (key, value) in s.items():
            if key == self.SECTION_NAME_KEY:
               pass
            elif value in (None, ''):
               tokens.append(str(key))
            else:
               tokens.append('%s:%s' % (key, value))
         sectionStrs.append(string.join(tokens))
      return string.join(sectionStrs, '|')

class Vowpal:
   '''Wrapper for Vowpal Wabbit machine learning classifier'''

   __slots__ = ('path_vw', 'path_models', 'path_cache', 'path_preds', 'path_data', 'vowpal_args')

   def __init__(self, path_vw='./vowpal_wabbit/vw', file_prefix='vw_models/test.%s', vowpal_args={}):
      self.path_vw = path_vw
      self.path_cache = file_prefix % 'cache'
      self.path_preds = file_prefix % 'preds'
      self.path_data = file_prefix % 'data'
      self.path_log = file_prefix % 'log'
      self.vowpal_args = vowpal_args

      for p in [self.path_cache, self.path_preds, self.path_data, self.path_log]: 
            if os.path.isfile(p):
                  os.remove(p)

   def predict(self, trainingRecords, testingRecords):
      self.writeRecords(trainingRecords + testingRecords)
      self.runVowpal()
      allPreds = self.readPreds()
      requestedPreds = {}
      for r in testingRecords:
         if str(r.id) in allPreds:
            requestedPreds[r.id] = allPreds[str(r.id)]
      return requestedPreds

   def writeRecords(self, records):
      f = open(self.path_data, 'a')
      for record in records:
            f.write(str(record) + '\n')
      f.close()

   def runVowpal(self):
      # these can be overriden using the vowpal_args constructor parameter
      argd = {
         '--conjugate_gradient' : None,
         '--passes' : '100', 
         '--regularization' : '.001',
         '--cache_file' : self.path_cache,
         '--predictions' : self.path_preds,
         '--data' : self.path_data
      }
      for (name, val) in self.vowpal_args.items():
         argd[name] = val
      argl = [self.path_vw]
      for (name, val) in argd.items():
         argl.append(str(name))
         if val != None:
            argl.append(str(val))
      log = open(self.path_log, 'w')
      p = subprocess.Popen(argl, stderr=subprocess.STDOUT, stdout=log)
      r = p.wait()
      log.close()
      if r != 0:
         raise Exception, ('Vowpal error occurred: check log file `%s`' % self.path_log)

   def readPreds(self):
      preds = {}
      for line in open(self.path_preds):
         (pred, id) = line.split()
         preds[id] = float(pred)
      return preds

if __name__ == '__main__':
   data = [
      [0.4, {'body' : {'height' : 0.8, 'weight' : 0.3}, 'age' : {'age' : 0.4}, 'sports' : { 'football' : None }}],
      [0.7, {'body' : {'height' : 0.8, 'weight' : 0.3}, 'age' : {'age' : 0.3}, 'sports' : { 'soccer' : None }}],
      [0.2, {'body' : {'height' : 0.8, 'weight' : 0.5}, 'age' : {'age' : 0.7}, 'sports' : { 'soccer' : None }}],
      [0.7, {'body' : {'height' : 0.9, 'weight' : 0.3}, 'age' : {'age' : 0.2}, 'sports' : { 'track' : None }}],
      [0.9, {'body' : {'height' : 0.7, 'weight' : 0.3}, 'age' : {'age' : 0.3}, 'sports' : { 'track' : None }}],
      [0.6, {'body' : {'height' : 0.7, 'weight' : 0.7}, 'age' : {'age' : 0.2}, 'sports' : { 'track' : None }}],
      [None, {'body' : {'height' : 0.7, 'weight' : 0.2}, 'age' : {'age' : 0.3}, 'sports' : { 'track' : None }}],
   ]
   records = []
   for i in xrange(len(data[:-1])):
      d = data[i]
      rc = VowpalRecord(i, d[0])
      for (namespace, section) in d[1].items():
         rc.addSection(namespace, section)
      records.append(rc)
   train = records[:-1]
   test = records[-1:]
   vw = Vowpal('./vowpal_wabbit/vw', './vw.%s', {'--passes' : '10' })
   preds = vw.predict(train, test)
   print 'preds are', preds

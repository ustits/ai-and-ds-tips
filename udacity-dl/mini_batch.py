example_features = [
  ['F11', 'F12', 'F13', 'F14'],
  ['F21', 'F22', 'F23', 'F24'],
  ['F31', 'F32', 'F33', 'F34'],
  ['F41', 'F42', 'F43', 'F44']]
# 4 Samples of labels
example_labels = [
  ['L11', 'L12'],
  ['L21', 'L22'],
  ['L31', 'L32'],
  ['L41', 'L42']]


def batches(batch_size, features, labels):
  assert len(features) == len(labels)
  result = []
  for i in range(0, len(features), batch_size):
    batch = [features[i: i + batch_size], labels[i: i + batch_size]]
    result.append(batch)
  return result


output = batches(3, example_features, example_labels)
print(output)

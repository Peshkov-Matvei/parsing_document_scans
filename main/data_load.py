from datasets import load_dataset

output_dir = 'data/funsd_layoutlmv3'

train_dataset = load_dataset('nielsr/funsd-layoutlmv3', split='train')
test_dataset = load_dataset('nielsr/funsd-layoutlmv3', split='test')

train_dataset.save_to_disk(f'{output_dir}/train')
test_dataset.save_to_disk(f'{output_dir}/test')

print(f'Dataset saved to {output_dir}')

# ECC显存校验

## ERR_TYPE 说明
- 0-7: 对应于8个比特位中的单个比特翻转
- 8: 所有位都可能翻转（随机多比特翻转）
- 9: ECC(64, 72) 编码后的错误注入
- 10-16: 部分比特保护（10保护1位，16保护7位）
- 17: 特殊编码 + ECC(64, 72) 汉明码
- 18: 特殊编码 + 稀疏汉明码
- 19: 模拟ECC(64, 50)行为，但不进行实际编码解码（Value-aware Parity Insertion ECC for Fault-tolerant Deep Neural Network)


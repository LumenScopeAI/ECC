# ECC显存校验

## ERR_TYPE 说明
- 0-7: 对应于8个比特位中的单个比特翻转
- 8: 所有位都可能翻转（随机多比特翻转）
- 9: ECC(64, 72) 编码后的错误注入
- 10-16: 部分比特保护（10保护1位，16保护7位）
- 17: 特殊编码 + ECC(64, 72) 汉明码 —— 参数+96偏移循环
- 18: 特殊编码 + 稀疏汉明码 —— 参数+96偏移循环保护前两位
- 19: 模拟ECC(64, 50)行为，但不进行实际编码解码（Value-aware Parity Insertion ECC for Fault-tolerant Deep Neural Network)





Value-aware Parity Insertion ECC for Fault-tolerant Deep Neural Network并不适用于LLM的参数分布，其大于阈值使用低位做校验位之后变0解码的操作会使得，此校验方法在高错误率时有优于ECC(64, 72)（9）和 参数+96偏移循环保护前两位（18）的表现，但是随着错误率降低，存在性能瓶颈，编解码操作带来的损失大于固有错误，且“稀疏矩阵”即额外存储空间开销极大，完全丧失优势。

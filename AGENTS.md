本项目为将 In-place Test Time Training 和 On-policy Distillation 结合的科研尝试。

documents/ 下为项目涉及的一些文档，例如：
- OPDTTT.md ：对项目原理的说明
- opdguide.md ：对项目执行方式的说明
- Inplace-TTT.pdf ：Inplace TTT 原论文
- In-Place Test-Time Training.html：原论文，html形式方便你查阅
- OPD.pdf ：OPD 原论文
- debug.md ：过去 debug 的记录
- guide.md ：用于纯 Inplace-TTT 复现的旧文件
- 其他各个文件

当我提出要求时，你需要按照我的要求维护其中部分文档使之为最新状态。

scripts/ 下为项目可能会用到的各种脚本。

VeOmni/ 是为本项目专门添加了兼容性层的一个第三方库。VeOmni/COMPATIBILITY_PATCHES.md 是兼容性补丁的说明（该说明为通用说明，不能包含任何 OPD-TTT 主项目的具体信息或绝对路径等）

本项目所用的环境为 .venv，每次执行脚本之前需要用 env_setup.sh 配置好环境，另外还有一个 .env 文件。

本服务器无法访问 huggingface 官网，你只能通过 hf-mirror 来访问。

log.txt 为大多数训练脚本的输出文件，所有训练脚本必须输出到日志文件中。如果选择log.txt作为输出文件，那么每次启动一个脚本时需要先清空log.txt

在你回答所有问题时，必须使用中文。
# 使用 miniconda 基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制 environment.yml 文件
COPY environment.yml /app/environment.yml

# 创建 conda 环境（移除 prefix 行避免路径问题）
RUN sed -i '/^prefix:/d' /app/environment.yml && \
    conda env create -f /app/environment.yml && \
    conda clean -afy

# 设置 conda 环境激活
SHELL ["conda", "run", "-n", "tsl", "/bin/bash", "-c"]

# 复制项目文件（仅复制存在的核心目录）
COPY data_provider/ /app/data_provider/
COPY exp/ /app/exp/
COPY layers/ /app/layers/
COPY scripts/ /app/scripts/
COPY utils/ /app/utils/

# 复制根目录 Python/Shell 文件
COPY *.py /app/
COPY *.sh /app/

# 创建运行时挂载点目录（dataset/models/config 等通过 -v 挂载）
RUN mkdir -p /app/models /app/config /app/LeaderBoard /app/logs /app/sh2yaml \
    /app/dataset /app/checkpoints /app/test_results /app/results /app/m4_results

# 设置默认的 conda 环境
ENV PATH=/opt/conda/envs/tsl/bin:$PATH
ENV CONDA_DEFAULT_ENV=tsl

# 设置入口点，确保在 tsl 环境中运行
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tsl"]

# 默认命令
CMD ["python", "run.py"]


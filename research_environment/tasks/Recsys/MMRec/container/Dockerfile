FROM continuumio/miniconda3
ENV DEBIAN_FRONTEND=noninteractive

# 配置代理 (构建完成后可删除这几行)
ARG http_proxy=http://127.0.0.1:7890
ARG https_proxy=http://127.0.0.1:7890
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    liblmdb-dev \
    g++ \
    libblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 复制环境文件
COPY environment.yml /app/environment.yml

# 2. 创建基础环境
RUN conda env create -f environment.yml || echo "Conda env partially created"

# RUN conda install -n mmrec -y lmdb -c conda-forge
RUN /opt/conda/envs/mmrec/bin/pip install --no-cache-dir lmdb==1.4.1

RUN conda run -n mmrec pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html && \
    conda run -n mmrec pip install torch-geometric==2.3.1

RUN conda run -n mmrec pip install sparsesvd==0.2.2 && \
    conda run -n mmrec pip install --upgrade setuptools

RUN conda clean -afy

# 3. 设置环境变量 - 使用 mmrec 环境 (修复: 之前错误地写成了 hc)
ENV PATH=/opt/conda/envs/mmrec/bin:$PATH
ENV CONDA_DEFAULT_ENV=mmrec
ENV CONDA_PREFIX=/opt/conda/envs/mmrec

# 4. 清除构建时的代理设置 (运行时不需要)
ENV http_proxy=""
ENV https_proxy=""

# 5. 复制项目代码
COPY . /app

# 6. 设置默认命令 - 直接使用 mmrec 环境的 bash
CMD ["/bin/bash"]
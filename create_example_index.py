import faiss
import numpy as np
import pickle
from datetime import datetime

def create_example_index():
    # 创建示例数据
    example_chunks = [
        {
            'type': 'section',
            'title': '奖学金政策',
            'content': '本科生奖学金申请时间为每年9月1日-15日，需在教务系统提交申请表。',
            'source': 'knowledge.txt',
            'created_at': datetime.now().isoformat()
        },
        {
            'type': 'section',
            'title': '实验室使用',
            'content': '主楼实验室开放时间为工作日8:00-21:00，使用前需在设备管理处登记。',
            'source': 'knowledge.txt',
            'created_at': datetime.now().isoformat()
        },
        {
            'type': 'section',
            'title': '校园卡补办',
            'content': '遗失校园卡请携带身份证到信息中心办理，补卡工本费20元。',
            'source': 'knowledge.txt',
            'created_at': datetime.now().isoformat()
        }
    ]
    
    # 创建示例向量（4096维的随机向量）
    num_chunks = len(example_chunks)
    vectors = np.random.rand(num_chunks, 4096).astype('float32')
    
    # 创建FAISS索引
    index = faiss.IndexFlatL2(4096)
    index.add(vectors)
    
    # 保存索引和映射
    index_data = {
        'index': index,
        'mapping': example_chunks
    }
    
    with open('data/faiss_index.pkl', 'wb') as f:
        pickle.dump(index_data, f)
    
    print("示例索引文件已创建：data/faiss_index.pkl")
    print("\n示例数据结构：")
    print("1. 索引维度：", index.d)
    print("2. 向量数量：", index.ntotal)
    print("3. 知识块数量：", len(example_chunks))
    print("\n知识块示例：")
    for chunk in example_chunks:
        print(f"\n标题：{chunk['title']}")
        print(f"内容：{chunk['content']}")
        print(f"类型：{chunk['type']}")
        print(f"来源：{chunk['source']}")
        print(f"创建时间：{chunk['created_at']}")

if __name__ == '__main__':
    create_example_index() 
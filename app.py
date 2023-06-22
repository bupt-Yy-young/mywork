from flask import Flask, render_template, request
import openai
import json
import numpy as np

app = Flask(__name__)

openai.api_key = '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'  # 替换为您的 OpenAI API 密钥

# 定义属性向量化函数和其他辅助函数
# ...

# 构建 prompt
prompt = """
Input: 用户输入一段角色的描述语句。
Output: 把这个描述语句转换为评分表。
输出模板如下：
{
    "test":{
    '攻击': 100(数值0-2000),
    '生命': 0(数值0-5000),
    '防御': 0(数值0-1000),
    '法抗': 0(数值0-100),
    '再部署时间': 70,
    '部署费用': 0,
    '攻击间隔': 0,
    '阻挡数': 0,
    '伤害类型': ['物理', '法术', '真实'],
    '伤害形式': ['近战', '远程'],
    '攻击范围': 0,
    '职业':['先锋', '术士','特种',"近卫",'重装','辅助','狙击','医疗'],
    '增幅能力': {'增加攻击': 0, '增加生命': 0, '增加防御': 0, '增加法抗': 0, '增加攻速': 0, '增加阻挡数': 0, '缩短再部署时间': 0,'降低费用':0},
    '削弱能力': {'降低攻击': 0, '降低防御': 0, '降低法抗': 0, '降低攻速': 0,'降低移速':0},
    '控制能力': {'停顿': 0, '眩晕': 0, '冻结': 0, '沉睡': 0, '沉默': 0, '束缚': 0, '浮空': 0,'失重':0,'位移':0},
    '辅助能力': {'加伤': 0, '减速': 0, '嘲讽': 0, '寒冷': 0, '隐匿': 0, '迷彩': 0, '承伤': 0,"庇护":0,"反隐":0,"抵抗":0,"闪避":0,"回复技力":0},
    '输出能力': {'物理': 0, '法术': 0, '真实': 0},
    '治疗能力': 0,
    '生存能力': 0,
    '回费能力':0
    }
}
下面是测试样例：
用户描述: 我需要一个攻击力高的，物理系的干员。

评分表: 
{
  "test": {
    "攻击": 2000,
    "生命": 0,
    "防御": 0,
    "法抗": 0,
    "再部署时间": 0,
    "部署费用": 0,
    "攻击间隔": 0,
    "阻挡数": 0,
    "伤害类型": ["物理"],
    "伤害形式": [""],
    "攻击范围": 0,
    "职业": [""],
    "增幅能力": {
      "增加攻击": 0,
      "增加生命": 0,
      "增加防御": 0,
      "增加法抗": 0,
      "增加攻速": 0,
      "增加阻挡数": 0,
      "缩短再部署时间": 0,
      "降低费用":0
    },
    "削弱能力": {
      "降低攻击": 0,
      "降低防御": 0,
      "降低法抗": 0,
      "降低攻速": 0,
      "降低移速":0
    },
    "控制能力": {
      "停顿": 0,
      "眩晕": 0,
      "冻结": 0,
      "沉睡": 0,
      "沉默": 0,
      "束缚": 0,
      "浮空": 0,
      "失重":0,
      "位移":0
    },
    "辅助能力": {
      "加伤": 0,
      "减速": 0,
      "嘲讽": 0,
      "寒冷": 0,
      "隐匿": 0,
      "迷彩": 0,
      "承伤": 0,
      "庇护":0,
      "反隐":0,
      "抵抗":0,
      "闪避":0,
      "回复技力":0
    },
    "输出能力": {
      "物理":90,
      "法术":0,
      "真实":0
    },
    "回费能力":0,
    "治疗能力":0,
    "生存能力":0
  }
}
"""

##在这输入干员文件
with open('character.json', 'r', encoding='utf-8') as file:
    hero_ratings = json.load(file)
#print(hero_ratings)


# 定义one-hot编码函数
def one_hot_encode(attr_value, attr_range):
    attr_vector = np.zeros(len(attr_range))
    if isinstance(attr_value, list):
        for attr_value_n in attr_value:
            if attr_value_n in attr_range:
                attr_vector[attr_range.index(attr_value_n)] = 1
    else:
        if attr_value in attr_range:
            attr_vector[attr_range.index(attr_value)] = 1
    #print(attr_value," ",attr_range," ",attr_vector)
    return attr_vector

# 定义属性向量化函数
def vectorize_attr(attr_dict):
    attr_vectors = []

    for attr_name,attr_value in attr_dict.items():
        #print(attr_value," ",attr_name)
        if isinstance(attr_value, int) or isinstance(attr_value, float):
            attr_vectors.append(attr_value)
        elif isinstance(attr_value, dict):
            #print(attr_value)
            for sub_attr_name in attr_value:
                attr_vectors.append(attr_value[sub_attr_name])
        #elif isinstance(attr_value, list):

            # print(attr_vectors)
            # print(attr_vectors)
        else:
            if(attr_name=='伤害类型'):
                attr_vectors.extend(one_hot_encode(attr_dict['伤害类型'], ['物理', '法术', '真实']))
            if(attr_name=='伤害形式'):
                attr_vectors.extend(one_hot_encode(attr_dict['伤害形式'], ['近战', '远程']))
            if(attr_name=='职业'):
                # print(0)
                attr_vectors.extend(one_hot_encode(attr_dict['职业'], ['先锋', '术士','特种',"近卫",'重装','辅助','狙击','医疗']))
                # print(one_hot_encode(attr_dict['职业'], ['先锋', '术士','特种',"近卫",'重装','辅助','狙击','医疗']))
            else:
                attr_vectors.append(0)
        #print(attr_vectors)
    # 将属性向量拼接起来
    #print(attr_vectors)
    return np.array(attr_vectors)


hero_attr_vectors = {}
for hero in hero_ratings:
    hero_attr_vectors[hero]=vectorize_attr(hero_ratings[hero])
# print(hero_attr_vectors)

def cos_sim(v1, v2):
    if len(v1) == 1:
        return v1[0]/v2[0]
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def cos_sim_neg(v1, v2):
    v3 = np.ones_like(v2)
    dot_product = np.dot(v1, v3)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v3)
    return dot_product / (norm_v1 * norm_v2)

def find_most_similar(candidate_vectors, target_vector,num_matches=3):
    #max_sim = -np.inf
    #most_similar_idx = -1
    res=[]
    #print(target_vector)
    neg = []
    pos = []
    neg_hero = {}
    pos_hero = {}
    for hero in candidate_vectors:
        neg_hero[hero] = []
        pos_hero[hero] = []
    for idx in range(len(target_vector)):
        if target_vector[idx] == 0:
            neg.append(target_vector[idx])
            for hero in candidate_vectors:
                neg_hero[hero].append(candidate_vectors[hero][idx])
        else:
            pos.append(target_vector[idx])
            for hero in candidate_vectors:
                pos_hero[hero].append(candidate_vectors[hero][idx])
    for hero in candidate_vectors:
        #print(candidate_vector)
        # sim = cos_sim(candidate_vectors[hero], target_vector)
        #print(i)
        # print(pos_hero[hero], pos)
        pos_sim = cos_sim(pos_hero[hero], pos)
        # print(pos_sim)
        neg_sim = cos_sim_neg(neg_hero[hero], neg)
        # print(neg_sim)
        sim = pos_sim*0.9+0.1*neg_sim
        res.append((hero,sim))
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)
    return res[:num_matches]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    q = request.form['description']
    result = evaluate(q)
    return render_template('result.html', result=result)

def evaluate(q):
    # 将您的已有代码放入此函数中，并根据用户输入的 q 返回结果
    # ...
    # 提交生成请求
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt}\n\n用户描述: {q}\n",
        max_tokens=1500,
        temperature=0.7
    )
    # 解析生成的响应
    output = response.choices[0].text.strip()
    # 分割字符串为行列表
    # 去掉开头的评分表
    json_text = output.split(':', 1)[1].strip()
    # 解析 JSON
    # print(json_text)
    data = json.loads(json_text)
    # 打印 JSON 对象
    print(data)
    user_rating = data

    attr_vector = vectorize_attr(user_rating['test'])

    # print(attr_vector)
    tmp = find_most_similar(hero_attr_vectors, attr_vector)
    # 返回结果
    return tmp

if __name__ == '__main__':
    app.run(debug=True)

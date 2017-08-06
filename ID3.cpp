/*****************************************************************
  ���ID3���������������������⣺�����������¶ȣ�ʪ�ȣ����޷硣
  ��Ԥ��һ���˻᲻��ȥ����
  ����ǵ�һ��д�ģ����Ա�������Ĵ��룬д����Щ���ҡ�
  ���Խ����
  sunny   cool    high    true
  The people maybe not go!
  rain    mild    normal  false
  The people maybe go!
  overcast hot    normal  false
  The people maybe go!

  By  ������
********************************************************************/
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <utility>
#include <cmath>
#include <map>
using namespace std;

vector<string>  attribute;                     //���Լ�
map<string,int> mp;                            //��ϣ��������Ӧ������attr_value����ĵ�ַ
vector<string>  attr_value[4];                 //ÿ�����Ե�ȡֵ

//���ݼ���
class Sample{
public:
    Sample(){};
    Sample(string, string, string, string, string);
    string          getAttributeValue(string);          //����һ�����Զ�Ӧ��ȡֵ
public:
    string          m_outlook;                          //��������
    string          m_temper;                           //�¶�����
    string          m_humidity;                         //ʪ������
    string          m_windy;                            //���޷�
    string          m_play;                             //ȥ��ȥ����

};

//IDE��������
class ID3{
public:
    typedef pair<string,ID3*> P;
    ID3(){}
    void            addSample(const Sample&);                        //���ڵ��������
    void            setAttribute(const vector<string>&);             //���ñ��ڵ����ʹ�õĻ�������
    double          calEntropy();                                    //���㱾�ڵ���Ϣ��
    int             getYesSample();                                  //����true�����ݸ���
    string          searchType(Sample);                       //�����������ԣ�Ԥ����
    void            divideTree();                                    //�ݹ�����ID3��
    void            eraseAttribute(const string&);                   //�����Լ��У�ɾ������ʹ�õ����ֵ�����
    double          getGain(const string&);                          //����һ�����Ե���Ϣ����

private:

    int             m_tag_leaf;                                      //��ǣ����ڵ��ǲ���Ҷ�ӽڵ�
    string          m_play;                                          //�����Ҷ�ӽڵ㣬����
    string          m_attribute;                                     //�������Ҷ�ӽڵ㣬���ڵ�Ļ�������
    vector<Sample>  m_sample_set;                                    //���ڵ�����ݼ�
    vector<P>       m_child;                                         //���ӽڵ��ָ��
    vector<string>  m_attribute_set;                                 //���ڵ����ʹ�õĻ������Լ�


};

//���ݼ����캯������ʼ��
Sample::Sample(string outlook,string temper,string humidity,
           string windy,string play) {
    m_outlook = outlook;
    m_temper  = temper;
    m_humidity = humidity;
    m_windy = windy;
    m_play = play;
}

//�������Զ�Ӧȡֵ
string  Sample::getAttributeValue(string value) {
    if(value=="outlook")
        return m_outlook;
    if(value=="temper")
        return m_temper;
    if(value=="humidity")
        return m_humidity;
    else
        return m_windy;
}

//���ñ��ڵ���ʹ�õĻ�������
void    ID3::setAttribute(const vector<string>& attr) {
    m_attribute_set = attr;
}

//�����Լ��У�ɾ�����νڵ�ʹ�õĻ�������
void    ID3::eraseAttribute(const string &attr) {

    vector<string>::iterator it = m_attribute_set.begin();
    while(it != m_attribute_set.end()) {
        if(*it == attr) {
            m_attribute_set.erase(it);
            break;
        }
        it++;
    }
}

//���ݸ������ԣ�Ԥ����
string  ID3::searchType(Sample sample) {
    ID3* p = this;
    while(!p->m_tag_leaf) {
        string attr_val = sample.getAttributeValue(p->m_attribute);
        for(vector<ID3>::size_type i = 0; i < p->m_child.size(); i++) {
            if(p->m_child[i].first == attr_val) {
                p = p->m_child[i].second;
                break;
            }
        }
    }
    return p->m_play;
}

//�������Ե���Ϣ����
double ID3::getGain(const string& attr) {
    double entropy = calEntropy();
    int index = mp[attr];
    vector<int> vec_num(attr_value[index].size(),0);
    vector<double> vec_entropy(attr_value[index].size(),0.0);

    for(string::size_type i = 0;i < attr_value[index].size(); i++) {
        vector<Sample> vec_sample;
        ID3* tree = new ID3();
        for(vector<Sample>::size_type j = 0;j < m_sample_set.size(); j++) {
            if(attr_value[index][i] == m_sample_set[j].getAttributeValue(attr)) {
                vec_sample.push_back(m_sample_set[j]);
                vec_num[i]++;
            }
        }
        if(!vec_sample.size()) {
            vec_entropy[i] = 0.0;
            continue;
        }else {
            for(vector<Sample>::size_type k = 0; k < vec_sample.size(); k++) {
                tree->addSample(vec_sample[k]);
            }
            vec_entropy[i] = tree->calEntropy();

        }

    }
    double _entropy = 0.0;
    for(vector<string>::size_type i = 0;i < attr_value[index].size(); i++) {
        _entropy += vec_entropy[i] * (double)vec_num[i] / m_sample_set.size();
    }

    return entropy - _entropy;
}

//�ݹ�����ID3��
void ID3::divideTree() {
    int play_yes = getYesSample();
    int play_no = m_sample_set.size()-play_yes;
    if(!attribute.size()) {
        if(play_yes>=play_no) {
            m_play = "yes";
        }
        else {
            m_play = "no";
        }
        m_tag_leaf = 1;
        return ;
    }
    if(!play_yes) {
        m_play = "no";
        m_tag_leaf = 1;
        return ;
    }
    if(!play_no) {
        m_play = "yes";
        m_tag_leaf = 1;
        return ;
    }
    m_tag_leaf = 0;
    string divide_attr;
    double gain = -1;
    for(string::size_type i = 0; i < m_attribute_set.size(); i++) {
        double temp_gain = getGain(m_attribute_set[i]);
        if(temp_gain > gain) {
            gain = temp_gain;
            divide_attr = m_attribute_set[i];
        }
    }

    m_attribute = divide_attr;
    eraseAttribute(divide_attr);
    int index = mp[divide_attr];

    for(vector<string>::size_type i = 0; i < attr_value[index].size(); i++) {
        ID3 *child = new ID3();
        child->m_attribute_set = m_attribute_set;
        vector<Sample> vec_sample;
        string divide_value = attr_value[index][i];

        for(vector<Sample>::size_type j = 0;j < m_sample_set.size(); j++) {
            if(divide_value == m_sample_set[j].getAttributeValue(divide_attr)) {
                vec_sample.push_back(m_sample_set[j]);
            }
        }
        if(!vec_sample.size()) {
            if(play_no <= play_yes) {
                child->m_play = "yes";
                child->m_tag_leaf = 1;
            }
            else {
                child->m_play = "no";
                child->m_tag_leaf = 1;
            }

            m_child.push_back(P(divide_value,child));
            return ;
        }
        else {
            m_child.push_back(P(divide_value,child));
            for(vector<Sample>::size_type k = 0;k < vec_sample.size(); k++) {
                child->addSample(vec_sample[k]);
            }
            child->divideTree();
        }
    }
}

//������������true��������
int ID3::getYesSample(){
        int yes= 0;
        for(vector<Sample>::size_type i = 0;i < m_sample_set.size(); i++) {
            if(m_sample_set[i].m_play == "yes"){
                yes++;
            }
        }
        return yes;
}

//�����ڵ��������
void ID3::addSample(const Sample &sample){
    m_sample_set.push_back(sample);
}

//���㱾�ڵ���Ϣ��
double ID3::calEntropy(){
    int _play_no = 0;
    int _play_yes = 0;
    for(vector<Sample>::size_type i = 0;i < m_sample_set.size(); i++) {
        if(m_sample_set[i].m_play == "no") {
            _play_no++;
        }
        else {
            _play_yes++;
        }
    }

    double _p_yes = (double)_play_yes / (m_sample_set.size());
    double _p_no = (double)1.0 - _p_yes;

    double tlog_no,tlog_yes;
    if(_p_no != 0.0) {
        tlog_no = _p_no*log(_p_no) / log(2);
    }
    else {
        tlog_no = 0.0;
    }
    if(_p_yes != 0.0) {
        tlog_yes = _p_yes * log(_p_yes) / log(2);
    }
    else {
        tlog_yes=0.0;
    }
    double res = -(tlog_yes+tlog_no);

    return res;
}

//��ʼ��ѵ�����ݼ�
void initialize() {
    attribute.push_back("outlook");
    attribute.push_back("humidity");
    attribute.push_back("temper");
    attribute.push_back("windy");
    mp["outlook"] = 0;
    attr_value[0].push_back("sunny");
    attr_value[0].push_back("overcast");
    attr_value[0].push_back("rain");
    mp["temper"] = 1;
    attr_value[1].push_back("hot");
    attr_value[1].push_back("mild");
    attr_value[1].push_back("cool");
    mp["humidity"] = 2;
    attr_value[2].push_back("high");
    attr_value[2].push_back("normal");
    mp["windy"] = 3;
    attr_value[3].push_back("false");
    attr_value[3].push_back("true");
}

int main()
{
    ID3 tree;
    cout<<"Please input the tarining set:"<<endl;
    initialize();
    for(int i=0;i<14;i++) {

        string outlook,temper,humidity,windy,play;
        cin >> outlook>>temper>>humidity;
        cin >> windy >> play;
        tree.addSample(Sample(outlook,temper,humidity,windy,play));


    }
    tree.setAttribute(attribute);

    tree.divideTree();
    int n;
    cout<<"Please input the number of test set:";
    cin >> n;
    for(int i=0;i<n;i++) {
        string outlook,temper,humidity,windy,play="";
        cin >> outlook >> temper >> humidity;
        cin >> windy;
        Sample sample(outlook,temper,humidity,windy,play);
        string res = tree.searchType(sample);
        if(res=="yes") {
            cout << "The people maybe go!" << endl;
        }else {
            cout << "The people maybe not go!" << endl;
        }
    }
    return 0;

}

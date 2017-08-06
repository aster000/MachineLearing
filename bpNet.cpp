/***********************************************************
    这个神经网络，用来解决异或问题。输入0或者1，来预
    测输出结果。训练5万次或者均方误差小于0.0001结束训练
    实际在3万次左右就能停止
    最后测试结果：
    0   0      0.007551
    0   1      0.993218
    1   0      0.993226
    0   0      0.007145
    结果不错 ^_^

    By 田宇坤
***********************************************************/

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
using namespace std;

const int  DATA_SIZE = 4;                    //输入数据集大小
const int  INPUT_LEVEL_SIZE = 2;             //输入层输入个数
const int  HIDE_LEVEL_SIZE = 2;              //隐层神经元数
const int  OUTPUT_LEVEL_SIZE = 1;            //输出层输出结果数

class BPNet{
public:
    BPNet();
    void        setData(double output[][OUTPUT_LEVEL_SIZE], double input[][INPUT_LEVEL_SIZE]);  //设置数据集
    double      predict(double,double);             //预测结果
    void        train();                            //训练
    void        print();                            //用于调试，查看参数
    void        printThresshlod();                  //用于调试，查看阈值

private:

    void        initialize();                        //初始化
    void        goForword(int);                      //前向传播
    void        changeDw(int);                       //调整隐层和输出层的权值
    void        changeDv(int);                       //调整输入层和隐层的权值
    void        calErrorRate();                      //计算均方误差
    void        changeDThressholdV(int);             //调整输出层神经元阈值
    void        changeDThressholdW(int);             //调整隐层神经元阈值
    double      getRand();                           //生成随机数（0,1）
    double      calG(int,int);                       //计算输出神经元梯度项
    double      calE(int,int);                       //计算隐层神经元梯度项
    double      sigmoid(double);                     //激活函数sigmoid



    double      m_output[DATA_SIZE][OUTPUT_LEVEL_SIZE];                    //数据集标记
    double      m_input[DATA_SIZE][INPUT_LEVEL_SIZE];                      //输入数据集
    double      m_temp_hide[HIDE_LEVEL_SIZE];                              //隐层神经元计算结果
    double      m_temp_output[DATA_SIZE][OUTPUT_LEVEL_SIZE];               //输出层输出
    double      m_i_h_weight[HIDE_LEVEL_SIZE][INPUT_LEVEL_SIZE];           //输入层与隐层权值
    double      m_h_o_weight[OUTPUT_LEVEL_SIZE][HIDE_LEVEL_SIZE];          //隐层与输出层权值
    double      m_threshold_hide[HIDE_LEVEL_SIZE];                         //隐层神经元阈值
    double      m_threshold_output[OUTPUT_LEVEL_SIZE];                     //输出层神经元阈值
    double      m_factor_hide;                                             //隐层学习率
    double      m_factor_output;                                           //输出层学习率
    double      m_error_rate;                                              //均方误差
};
//构造函数初始化
BPNet::BPNet() {
    initialize();
}

//打印阈值，用于调试
void    BPNet::printThresshlod() {
    cout<<"hide1:" <<m_threshold_hide[0]<<endl;
    cout<<"hide2:" <<m_threshold_hide[1] <<endl;
    cout<<"output:"<<m_threshold_output[0]<<endl;
}

//生成随机函数
double  BPNet::getRand() {

    double res = rand() % 10;
    res = res / 11.0 + 0.1;
    if(res == 1.0) {
        res /= 2.0;
    }

    return res ;
}

//计算均方误差
void    BPNet::calErrorRate() {
    m_error_rate = 0.0;
    double temp = 0.0;
    for(int i = 0; i < DATA_SIZE; i++) {
        temp = fabs(m_output[i][0] - m_temp_output[i][0]);
        m_error_rate += temp*temp;
    }
    m_error_rate /= (double)2.0;
    cout<<"The error rate is:"<<m_error_rate<<endl;
}

//训练
void    BPNet::train(){
    for(int j = 0; j< 50000 && m_error_rate > 0.0001; j++){
        cout<<"Start the "<<j<<" training!"<<endl;
        for(int i = 0; i < DATA_SIZE; i++) {
            goForword(i);
            changeDw(i);
            changeDThressholdW(i);
            changeDv(i);
            changeDThressholdV(i);
        }
        calErrorRate();
    }
}

//预测结果
double  BPNet::predict(double num1,double num2) {
    for(int i =0 ;i < HIDE_LEVEL_SIZE; i++) {
        m_temp_hide[i] = num1 * m_i_h_weight[i][0] + num2 * m_i_h_weight[i][1];
        m_temp_hide[i] = sigmoid(m_temp_hide[i]-m_threshold_hide[i]);
    }
    double result = 0.0;
    for(int i = 0; i < HIDE_LEVEL_SIZE; i++) {
        result += m_temp_hide[i] * m_h_o_weight[0][i];
    }
    return      sigmoid(result-m_threshold_output[0]);
}

//计算输出层梯度项
double  BPNet::calG(int dataID,int j) {
    double y1 = m_temp_output[dataID][j];
    double y =  m_output[dataID][j];
    return y1 * (1.0 - y1)*(y - y1);
}

//计算隐层梯度项
double  BPNet::calE(int dataID,int h) {
    double temp = m_temp_hide[h] * (1.0 - m_temp_hide[h]);
    double res = 0.0;
    for(int i = 0; i < OUTPUT_LEVEL_SIZE; i++) {
        double   g_j = calG(dataID,0);
        res += temp * m_h_o_weight[i][h]*g_j;
    }
    return res;
}

//调整隐层与输出层权值
void    BPNet::changeDw(int dataID) {

    for(int j = 0; j < OUTPUT_LEVEL_SIZE; j++) {
        double g_j = calG(dataID, j);
        for(int h = 0; h < HIDE_LEVEL_SIZE; h++) {
            double dw = m_factor_output * g_j * m_temp_hide[h];
            m_h_o_weight[j][h] += dw;
        }
    }
}

//调整输入层与隐层权值
void    BPNet::changeDv(int dataID) {
    for(int h = 0; h < HIDE_LEVEL_SIZE; h++) {
        double e_h = calE(dataID,h);
        for(int i = 0; i < INPUT_LEVEL_SIZE; i++) {
            double dv = m_factor_hide * e_h * m_input[dataID][i];
            m_i_h_weight[h][i] += dv;
        }
    }
}

//调整输出层神经元阈值
void    BPNet::changeDThressholdW(int dataID) {
    for(int j = 0; j < OUTPUT_LEVEL_SIZE; j++) {
        double  g_j = calG(dataID,j);
        double  d_w = -m_factor_output * g_j;
        m_threshold_output[j] += d_w;
    }
}

//调整隐层神经元阈值
void    BPNet::changeDThressholdV(int dataID) {
    for(int h = 0; h < HIDE_LEVEL_SIZE; h++) {
        double  e_h = calE(dataID,h);
        double  d_v = -m_factor_hide * e_h;
        m_threshold_hide[h] += d_v;
    }
}

//前向传播
void    BPNet::goForword(int dataID) {
    for(int i = 0; i < HIDE_LEVEL_SIZE; i++) {
        m_temp_hide[i] = 0.0;
        for(int j = 0; j < INPUT_LEVEL_SIZE; j++) {
            m_temp_hide[i] += m_input[dataID][j]*m_i_h_weight[i][j];
        }

        m_temp_hide[i] = sigmoid(m_temp_hide[i]-m_threshold_hide[i]);

    }
    for(int i = 0; i < OUTPUT_LEVEL_SIZE; i++) {
        m_temp_output[dataID][i] = 0.0;
        for(int j = 0; j < HIDE_LEVEL_SIZE; j++) {
            m_temp_output[dataID][i] += m_temp_hide[j] * m_h_o_weight[i][j];
        }
        m_temp_output[dataID][i] = sigmoid(m_temp_output[dataID][i] - m_threshold_output[i]);
    }


}

//打印，用于调试
void    BPNet:: print() {
    for(int i = 0; i < DATA_SIZE; i++) {
        cout<<m_input[i][0]<<"xor"<<m_input[i][1]<<"="<<m_output[i][0]<<endl;
    }
}

//初始化
void    BPNet::initialize(){
    for(int i = 0; i < HIDE_LEVEL_SIZE; i++) {
        m_threshold_hide[i] = getRand();
    }
    for(int i = 0; i< OUTPUT_LEVEL_SIZE; i++) {
        m_threshold_output[i] = getRand();
    }

    m_factor_hide = 0.3;
    m_factor_output = 1.0;
    m_error_rate = 1.0;
    srand((unsigned)time(NULL));
    for(int i = 0; i < HIDE_LEVEL_SIZE; i++) {
        for(int j = 0; j< INPUT_LEVEL_SIZE; j++) {
            m_i_h_weight[i][j] = getRand();

        }
    }
    for(int i = 0; i < OUTPUT_LEVEL_SIZE; i++) {
        for(int j = 0; j < HIDE_LEVEL_SIZE; j++) {
            m_h_o_weight[i][j] = getRand();
        }
    }
}

//激活函数sigmoid
double  BPNet::sigmoid(double x) {
    return  1.0 / (1 + exp(-x) );
}

void    BPNet::setData(double output[][OUTPUT_LEVEL_SIZE],
            double input[][INPUT_LEVEL_SIZE]) {
    for(int i = 0; i < DATA_SIZE; i++) {
        for(int j = 0; j < OUTPUT_LEVEL_SIZE; j++) {
            m_output[i][j] = output[i][j];
        }
    }
    for(int i = 0; i< DATA_SIZE; i++) {
        for(int j = 0; j < INPUT_LEVEL_SIZE; j++) {
            m_input[i][j] = input[i][j];
        }
    }
}

int main()
{
    double     output[][OUTPUT_LEVEL_SIZE] = {0.0,1.0,1.0,0.0};
    double     input[][INPUT_LEVEL_SIZE] = {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0};
    BPNet      bp_net;
    bp_net.setData(output,input);
    bp_net.train();

    for(int i=0;i<4;i++) {
        double num1,num2;
        cin >> num1 >> num2;
        cout<<bp_net.predict(num1,num2)<<endl;
    }

    return 0;
}

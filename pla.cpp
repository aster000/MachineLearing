#include <iostream>
#include <vector>
using namespace std;

class Vec{
public:
    friend double operator*(const Vec&,const Vec&);
    Vec() = default;
    Vec(int len) {
        for(int i=0;i!=len;i++)  {
            m_vec.push_back(1);
        }
    }
    Vec(const vector<int>& v):m_vec(v){}
    int& operator[](int i)  {
        return m_vec[i];
    }
    const int& operator[](int i) const{
        return m_vec[i];
    }
    int size() const {
        return m_vec.size();
    }
    Vec& operator+=(const Vec&x) {
        for(int i=0;i!=x.size();i++) {

            (*this)[i] += x[i];
        }
        return *this;
    }

private:
    vector<int>  m_vec;

};
class PLA{
public:
    PLA() = default;
    PLA(const Vec& w,const Vec& y,const vector<Vec>& train):m_w(w),m_y(y),m_train(train){}
    void train();
    bool predict(const Vec& x);


private:
    int sign(double x);
    Vec     m_w;
    Vec     m_y;
    vector<Vec>  m_train;

};
int PLA::sign(double x) {
    if(x >=0)
        return 1;
    else
        return -1;
}
void PLA::train(){
    while(true) {
        bool flag = 1;
        for(int i=0;i<m_y.size();i++) {
            if(m_y[i]!=sign(m_w*m_train[i])) {
                flag = 0;
                for(int j;j!=m_train.size();j++) {
                    m_train[i][j] *= m_y[i];
                }
                m_w +=  m_train[i];
                break;

            }
        }
        if(flag){
            break;
        }
    }
}
bool PLA::predict(const Vec& x) {
    return sign(m_w*x);
}
double operator*(const Vec&w, const Vec&x) {
    double res = 0.0;
    for(int i = 0; i != w.size(); i++) {

        res += w[i] * x[i];
    }
    return res;
}
int main()
{

    return 0;
}

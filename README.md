# Leetcode-practice

## LRU缓存

### 使用哈希表和双向链表实现 利用哈希表查找复杂度和链表的添加删除复杂度

```c++
class LRUCache {
private:
    int capacity;
    list<pair<int,int>> cache;
     unordered_map<int, list<pair<int, int>>::iterator> map;    
public:
    LRUCache(int capacity) {
        this->capacity=capacity;
    }
    
    int get(int key) {
        auto it=map.find(key);
        if(it == map.end()) return -1;
        pair<int,int> ans= *map[key];
        cache.erase(map[key]);
        cache.push_front(ans);
        map[key]= cache.begin();
        return ans.second;
    }
    
    void put(int key, int value) {
        auto it=map.find(key);
        if(it == map.end()){
            if(map.size()==capacity){
                auto temp=cache.back();
                map.erase(temp.first);
                cache.pop_back();
            }
            cache.push_front(make_pair(key,value));
            map[key]=cache.begin();
            return;
        }
        cache.erase(map[key]);
        cache.push_front(make_pair(key,value));
        map[key]= cache.begin();
    }
};
```

## 三数相加为0

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int N = nums.size();
        vector<vector<int> > res;
        for (int i = 0; i < N - 2; ++i) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1;
            int r = N - 1;
            while (l < r) {
                int s = nums[i] + nums[l] + nums[r];
                if (s > 0) {
                    --r;
                } else if (s < 0) {
                    ++l;
                } else {
                    res.push_back({nums[i], nums[l], nums[r]});
                    while (l < r && nums[l] == nums[++l]);
                    while (l < r && nums[r] == nums[--r]);
                }
            }
        }
        return res;
    }
};
```



## String的乘法

### 基于竖乘原则来进行编程 降低复杂度

```c++
class Solution {
public:
    string multiply(string num1, string num2) {
        int length1=num1.size(),length2=num2.size();
        string res(length1+length2,'0');

        for(int i=length1-1; i>=0; --i){
            for (int j=length2-1; j>=0; --j){
                int temp=res[i+j+1]-'0'+(num1[i]-'0')*(num2[j]-'0');
                res[i+j+1]='0'+temp%10;
                res[i+j]+=temp/10;
            }
        }

        for(int i=0; i<res.size();++i){
            if (res[i]-'0') return res.substr(i);
        }

        return "0";
    }
};
```



## 缺失的第一个正数

### 通过记录1~size内最小未出现的正整数实现

#### 原版本

```C++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        vector<int> vec(nums.size()+2);
        for(int i=0; i<nums.size() ;++i){
            if(nums[i]>=0 && nums[i]<= nums.size()) vec[nums[i]]=1;
        }
        for(int i=1; i<vec.size(); ++i){
            if(vec[i]==0) return i;
        }
        return 0;
    }
};
```

#### 新版本

```c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int length=nums.size();
        for(int index=0; index <length ; ++index){
            while(nums[index]>0 && nums[index]<=nums.size() && nums[index]!=index+1 && nums[nums[index]-1]!=nums[index]){
                swap(nums[index],nums[nums[index]-1]);
            }
        }
        for(int index=0; index< nums.size();++index){
            if(index+1!=nums[index]) return index+1;
        }
        return length+1;
    }
};
```

##### 

## 跳跃游戏

#### 原版本

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int length=nums.size();
        vector<int> flag(length);
        for(int i=0; i<length; ++i){
            if(flag[i] || i==0 ) {for(int j=0; j<=nums[i] && j+i<length; ++j) flag[j+i]=1;continue;}
            break;
        }
        for(int i=1; i<length; ++i){
            if(flag[i]==0) return false;
        }
        return true;
    }
};
```

#### 新版本

##### 最远可到达的点进行比较

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
    int loc= 0;
	for (int temp = 0; temp < nums.size(); ++temp)
	{
		if (temp > loc) return false;
		loc = max(loc, temp + nums[temp]);
        if (loc >=nums.size()) return true;
	}
	return true;
    }
};
```



## 删除倒数第N个节点

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *last_node=head;
        ListNode *temp_node=head;
        ListNode *prev_node=nullptr;
        while(n-1){
            --n;
            last_node=last_node->next;
        }
        while(last_node->next){
            prev_node=temp_node;
            last_node=last_node->next;
            temp_node=temp_node->next;
        }
        if(head==temp_node) return head->next;
        prev_node->next=temp_node->next;
        return head;
    }
};
```


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



## 盛水最多容器

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i=0,j=height.size()-1;
        int max_area=(j-i)*min(height[i],height[j]);
        int flag=0;
        for(;i<j;){
            max_area=max(max_area,min(height[i],height[j])*(j-i));
            if(height[j]>height[i]) ++i;
            else --j;
        }
    return max_area;
    }
};
```



## 字符串中的查找与替换

```c++
class Solution {
public:
    string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
        int length=S.size();
        string res="";
        vector<int> flag(length,-1);
        int com_len=indexes.size();
        for(int i=0; i<com_len; ++i){
            int size=sources[i].size();
            string sub=S.substr(indexes[i],size);
            if(sources[i].compare(sub)==0) {
                flag[indexes[i]]=i;
                while(size!=1){
                    flag[indexes[i]+(--size)]=-2;
                }
            }
        }
        for(int index=0; index<length; ++index){
            if(flag[index]==-1){res+=S[index];}
            if(flag[index]>=0){res+=targets[flag[index]];}
        }
        return res;
    }
};
```



## 常数时间插入、删除和获取随机元素

hashmap用来 key->index(vector的索引位置),vector用来存储key值。插入比较简单，使用hashmap是映射索引便于在O(1)时间可以删除vector所对应的元素。

```c++
class RandomizedSet {
private:
    unordered_map<int,int> mymap;
    vector<int> key_index;
public:
    /** Initialize your data structure here. */
    RandomizedSet() {
        
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val) {
        auto it=mymap.find(val);
        if(it != mymap.end()) return false;
        mymap[val]=key_index.size();
        key_index.push_back(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val) {
        auto it=mymap.find(val);
        if(it == mymap.end()) return false;
        int size=key_index.size();
        int loc=mymap[val];

        mymap[key_index[size-1]]=loc;
        key_index[loc]=key_index[size-1];
        key_index.pop_back();
        mymap.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        int size=key_index.size();
        if(size == 0) return 0;
        int index=rand()%size;
        return key_index[index];
    }
};
```



## 环形链表Ⅱ

使用快慢指针进行解决（相遇点距起点距离必定为环大小的整数倍），从相遇点和起点分别开始再走，相遇时必定为环起点

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow=head;
        ListNode *fast=head;
        ListNode *res=head;
        while(fast && fast->next){
            fast=fast->next->next;
            slow=slow->next;

            if(fast==slow){
                while(res != slow){
                    slow=slow->next;
                    res=res->next;
                }
                return res;
            }
        }
        return nullptr;
    }
};
```



## 两数相加

```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head= new ListNode(0);
        ListNode *res=head;
        int add_flag=0;
        int val1,val2;
        while(l1 || l2 || add_flag){
            int val1=0,val2=0;
            if(l1) {val1=l1->val;l1=l1->next;}
            if(l2) {val2=l2->val;l2=l2->next;}
            ListNode *tempNode=new ListNode((add_flag+val1+val2)%10);
            head->next=tempNode;
            head=head->next;
            add_flag=(add_flag+val1+val2)/10;
        }
        return res->next;
    }
};
```



## 无重复字符的最长子串

滑动窗口

```c++
int lengthOfLongestSubstring(string s) {
    int max=0;
    string subs=s.substr(0,1);
    if(s.size()==1) return 1;
    for(int i =1;i<s.size();++i){
        auto pos=subs.find(s[i]);
        if (pos==-1) {subs.append(s,i,1);continue;}
        max=(subs.size()>max)? subs.size():max;
        subs.erase(0,pos+1);
        subs.append(s,i,1);
    }
    max=(subs.size()>max)? subs.size():max;
    return max;
    }
```

哈希表

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> mymap;
        int max=mymap.size();
        int length=s.size();

        for(int index=0; index<length; ++index){
            auto it=mymap.find(s[index]);
            if(it != mymap.end()) {
                int val=mymap[s[index]];
                while(val && mymap.find(s[--val])!=mymap.end() && val==mymap[s[val]]){
                    mymap.erase(s[val]);
                }
            }
            mymap[s[index]]=index;
            max=(max>mymap.size())? max:mymap.size();
        }
        return max;
    }
};
```



## 最大回文子串

Manacher算法 O(N)复杂度寻找最长回文子串

 Given a string `s`, return *the longest palindromic substring* in `s`. 

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int length=2*s.size();
        int max_size=0;
        string res;
        for(int index=1; index<length; ++index){
            int temp_shift=0;
            while(index-temp_shift>=0 && index+temp_shift<=length){
                if((index-(temp_shift))%2==0 || s[(index-(temp_shift))/2]==s[(index+(temp_shift))/2]) {++temp_shift;continue;}
                break;
            }
            --temp_shift;
            if(temp_shift>max_size){
                max_size=temp_shift;
                res=s.substr((index-temp_shift)/2,temp_shift);
            }
        }
        return res;        
    }
};
```

## 树

### 遍历

#### 题目：

 给定一个二叉树，将它展开为一个单链表。 

[Leedcode链接]: https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/



#### 代码：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        static vector<TreeNode*> vec;
        if (root == nullptr) return;
        TreeNode* left=root->left;
        TreeNode* right=root->right;
        root->left=nullptr;
        if(left){
            root->right=left;
            if(right) vec.push_back(right);
        }
        while(root->left==nullptr && root->right==nullptr && vec.size()){
            root->right=vec.back();
            vec.pop_back();
        }
        flatten(root->right);
    }
};
```


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



## 链表

### 单向链表

#### 题目：[相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

##### 题目描述：

编写一个程序，找到两个单链表相交的起始节点。

##### 题目要求：

- 如果两个链表没有交点，返回 nullptr
- 在返回结果后，两个链表仍须保持原有的结构
- 可假定整个链表结构中没有循环
- 程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存

##### 解题思路：

首先介绍暴力方法，题目要求查找两个单链表的相交的起始节点，直观上，我们利用set容器存入链表中各个节点的信息，该容器可以用来判断节点出现的情况，然后遍历另外一条链表中节点判断是否出现过即可。这种方式可以满足O(m+n)的时间复杂度，但不能满足O(1)的内存需求。

其次介绍更为精妙的方法，两条链表如果相同长度的情况下，如何判断是否两条单链表相交的起始点，方法很简单，通过双指针法即可。具体如下，定义两个指针分别指向两个head，然后不断判断是否相等，不断向后走即可。但是该题目中，两条链表不是相同长度，解决了该问题，那问题就非常简单。如果我们把两条链表连起来（不是真正的连，只是到尾部时，跳转到另外一条链表的头部），那长度就是相同的了，并且通过这种找到的节点也是相交起始点，因为相交的部分与原问题完全相同。具体实现方法是定义双指针，指向两条链表的头部，然后比较，同步后移，只是每条到达底部时跳转至另外的头部（每个指针只跳转一次即可），继续比较，后移。

##### 代码：

暴力方法：

```c++
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode * res=nullptr;
    unordered_set<ListNode *> _ptrset;
    while(headA){
        _ptrset.emplace(headA);
        headA=headA->next;
    }
    while(headB){
        auto _findIter=_ptrset.find(headB);
        if(_findIter != _ptrset.end()) {res=*_findIter;break;}
        headB=headB->next;
    }
    return res;
}
```

 双指针方法：

```c++
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
ListNode *ptrA=headA,*ptrB=headB;
int flagA=1,flagB=1;
while(ptrA && ptrB && ptrA!=ptrB){
ptrA=(ptrA->next==nullptr && flagA--)? headB:ptrA->next;
ptrB=(ptrB->next==nullptr && flagB--)? headA:ptrB->next;
}
if(ptrA==ptrB) return ptrA;
else return nullptr;
}
```



### 环形链表

#### 环形链表

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

## 动态规划

#### 题目：[最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

##### 问题描述：

 给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。题目的问题是求解以下问题
$$
\max_{StartIndex \leq i \leq EndIndex}{f(i)}
$$
 f(i)表示以node i结尾的最大子序列和。在遍历至元素i+1时，
$$
f(i+1)=\max\{f(i)+nums[i+1],nums[i+1]\}
$$
最后引入个变量来记录最优的f(i)即可

##### 解题思路：

题目要求实现一个O(N)算法，因此不能使用暴力的进行求解。因此目标是一次遍历就寻找出最优的解。

##### 代码：

```c++
int maxSubArray(vector<int>& nums) {
    int pre = 0, maxres = nums[0];
    for (const auto &x: nums) {
        pre = max(pre + x, x);
        maxres = max(maxres, pre);
    }
    return maxres;
}
```

## 数组

#### 题目：[ 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

##### 问题描述：

给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

##### 要求：

-  不要使用除法，在 O(*n*) 时间复杂度内完成
- 在常数空间复杂度内完成？（ 出于对空间复杂度分析的目的，**输出数组不被视为**额外空间。）

##### 解题思路：

该题目需要计算除开自身元素外，其他元素的乘积，一个直观的方法便是将所有元素相乘，对于每个元素相除即可，也能满足其复杂度需求。但该种解决方法存在问题，若输入数组中存在0元素，还需要很多额外讨论。因此题目中要求了不允许使用除法运算。最简单的方法便是，对于每个元素，计算其他元素的乘积即可，但这种方式时间复杂度为O(N^2)，不能满足要求，所以首先第一步需要分析各个点最后结果之间的关系，如何利用遍历快速得到结果。



![Image text](pic/productExceptSelf1.png)

对于任何节点的返回值，都可以分为两部分，在该节点左侧元素乘积*右侧元素乘积，如上图所示。

![Image text](pic/productExceptSelf2.png)

上图中，列出了返回结果中具体的因子项，从上图中可以观察而出，通过index从0至n-2相乘遍历可以计算出返回结果vector中各个需要的左侧乘积，从n-1至2遍历可计算出所有需要的右侧乘积。最后剩下的问题只有常数空间的问题，在右侧乘积遍历时，因为一个迭代的中间量即可，然后实现右侧元素乘积的记录与结果的更新。

##### 代码：

```c++
vector<int> productExceptSelf(vector<int>& nums) {
    int n=nums.size();
    vector<int> res(n,1);
    for(int i{1};i<n;++i){
        res[i]=res[i-1]*nums[i-1];
    }
    int R=1;
    for(int j{n-2};j>=0;--j){
        R*=nums[j+1];
        res[j]*=R;
    }
    return res;
}
```


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # len1 = 0
        # len2 = 0
        # t1 = l1
        # t2 = l2
        # while(t1 != None):
        #     len1 +=1
        #     t1 = t1.next
        # while(t2 != None):
        #     len2 += 1
        #     t2 = t2.next
        t1 = l1
        t2 = l2
        # min_len = len1 if len1 <= len2 else len2

        remains = [0]
        result = None

        while (t1 is not None or t2 is not None):
            r = ListNode(0, None)
            total = 0
            if t1 is None:
                total = t2.val
            elif t2 is None:
                total = t1.val
            else:
                total = t1.val + t2.val
            print(total)
            if total >= 10:
                remain = total // 10
                total = total % 10
                remains.append(remain)

            r.val = total + remains[-1]
            #print(r.val)
            t1 = t1.next
            t2 = t2.next
            if result is None:
                result = r
            else:
                result.next = r
        print(remains)
        return result

a = [2,4,3]
b = [5,6,4]
l1 = None
l2= None
for i in a:
    r = ListNode(i, None)
    if l1 is None:
        l1 = r

    else:
        l1.next = r
for i in b:
    r = ListNode(i, None)
    if l2 is None:
        l2 = r
        t = l2
    else:
        t = t.next
        t = r

# result = Solution()
# r = result.addTwoNumbers(l1,l2)
# t = r
# while( t is not None):
#     print(t.val)
#     t = t.next
t2 = l2
while( t2 is not None):
    print(t2.val)
    t2 = t2.next

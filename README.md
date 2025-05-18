## **8-Puzzle Project – Báo cáo tóm tắt**
______________
**1. Mục tiêu**

Mục tiêu của đồ án là áp dụng các thuật toán trí tuệ nhân tạo để giải bài toán 8-Puzzle (trò chơi 8 ô chữ). Bài toán 8-Puzzle gồm một bàn cờ 3×3 có 8 ô chứa số và 1 ô trống (ký hiệu 0), nhiệm vụ là tráo các ô để từ trạng thái bắt đầu đạt đến trạng thái mục tiêu. Các thuật toán AI được áp dụng bao gồm tìm kiếm không có thông tin, tìm kiếm có thông tin (heuristic), tìm kiếm cục bộ, tìm kiếm trong môi trường phức tạp, tìm kiếm có ràng buộc và học tăng cường. Trạng thái (state) là một cấu hình cụ thể của bàn 3×3, mục tiêu (goal) là cấu hình cuối (ví dụ {1,2,3;4,5,6;7,8,0}), còn lời giải (solution) là chuỗi các bước di chuyển ô trống (L/R/U/D) đưa trạng thái đầu đến đích. Bài toán này có độ phức tạp tăng theo cấp số mũ với kích thước bảng, mỗi trạng thái có tối đa 4 hành động khả dĩ.
___________
**2. Nội dung**

**2.1 Tìm kiếm không có thông tin (BFS, DFS, UCS, IDS)**

**Giới thiệu:** Nhóm thuật toán tìm kiếm không có thông tin (blind search) quét không gian trạng thái theo quy tắc nhất định mà không sử dụng thêm kiến thức của bài toán (không có heuristic). Khi áp dụng cho 8-Puzzle, các thành phần chính vẫn là: trạng thái là cấu hình bàn 3×3, đích là trạng thái mục tiêu, lời giải là chuỗi di chuyển L,R,U,D.

**•	BFS (Breadth-First Search):** Duyệt theo chiều rộng, xử lý lần lượt các trạng thái ở cùng cấp độ. BFS đảm bảo tìm được lời giải có số bước di chuyển tối thiểu (ngắn nhất), nhưng đổi lại yêu cầu bộ nhớ rất lớn vì phải lưu nhiều trạng thái đợi xử lý.

**•	DFS (Depth-First Search):** Duyệt theo chiều sâu, ưu tiên mở rộng nhánh sâu nhất. DFS tiêu thụ ít bộ nhớ hơn BFS nhưng không đảm bảo tìm ra lời giải ngắn nhất hoặc thậm chí có thể không tìm được nếu bị lặp vô hạn. DFS dễ bị mắc kẹt ở các nhánh sâu (độ sâu giới hạn <em>max_depth</em>) và chỉ tìm lời giải khi vét hết một nhánh.

**•	UCS (Uniform-Cost Search):** Thực chất là phiên bản tổng quát của Dijkstra, mở rộng nút có tổng chi phí (số bước) nhỏ nhất trước. UCS luôn tìm ra đường đi chi phí thấp nhất (tương đương ngắn nhất khi mỗi bước có cost=1). Đổi lại, chi phí tính toán có thể lớn tương tự BFS nhưng đảm bảo độ tối ưu của lời giải.

**•	IDS (Iterative Deepening Search):** Kết hợp ưu điểm của BFS và DFS. IDS thực hiện nhiều lần DFS với độ sâu tăng dần, bắt đầu từ 0 đến một ngưỡng. Phương pháp này vừa đảm bảo tìm lời giải ngắn nhất vừa sử dụng bộ nhớ thấp (của DFS). Mỗi lượt lặp, IDS thực hiện DFS có giới hạn độ sâu, cho đến khi tìm thấy đích.

**Hình ảnh GIF minh họa:**
**- BFS:**

![BFS](https://github.com/user-attachments/assets/af9876c2-948d-4500-820a-5f831c3b564b)

**- DFS:**

![DFS](https://github.com/user-attachments/assets/4e774b8c-320b-420a-b4de-5b88f3e6a4ce)

**- UCS:**

![UCS](https://github.com/user-attachments/assets/41950b41-968d-4457-8472-bf2652bba14c)

**- IDS:**

![IDS](https://github.com/user-attachments/assets/89b982cb-2d76-4ccc-abd0-34d2122f9de6)

Hiệu suất: Nhóm thuật toán này đều có tính toàn diện (completeness) trong các môi trường vô hướng (8-puzzle đảm bảo có lời giải hoặc báo vô nghiệm). BFS/UCS cho lời giải tối ưu nhưng bị giới hạn bộ nhớ và thời gian tăng theo bậc, vì số trạng thái tăng cực nhanh với độ sâu (độ phức tạp thời gian O(b^d) với b~4, d là độ sâu). DFS ít tốn bộ nhớ nhưng có thể tìm ra lời giải rất không tối ưu hoặc không tìm được do lặp vô hạn. IDS khắc phục nhược điểm này, mất thời gian nhưng bộ nhớ vẫn hạn chế.

2.2 Tìm kiếm có thông tin (Greedy, A*, IDA*)

**2.2 Tìm kiếm có thông tin (Greedy, A_Star, IDA_Star**
**Giới thiệu:** Nhóm tìm kiếm có thông tin sử dụng hàm heuristic (ước lượng chi phí còn lại) để dẫn dắt quá trình tìm kiếm. Trong bài 8-Puzzle, heuristic thường dùng là: số ô sai vị trí (misplaced tiles) hoặc tổng khoảng cách Manhattan. Các thành phần: trạng thái, mục tiêu, solution giống trên, nhưng khi đánh giá các trạng thái mở rộng, thuật toán sẽ dùng hàm ước lượng h(n).

**•	Greedy Best-First Search:** Là tìm kiếm theo heuristic thuần tuý. Mỗi bước, thuật toán mở rộng trạng thái có giá trị heuristic nhỏ nhất (h(n) – ước lượng còn lại đến đích). Thuật toán này tìm đường đi rất nhanh nhưng không đảm bảo tối ưu, có thể rẽ vào ngõ cụt hoặc mắc kẹt ở cực tiểu cục bộ.

**•	A_Star:** Kết hợp BFS và Greedy. Đánh giá theo hàm f(n)=g(n)+h(n), trong đó g(n) là chi phí đã đi được (số bước), h(n) là ước lượng (Manhattan). Với heuristic khả tín (admissible), A* luôn tìm đường ngắn nhất. Đây là thuật toán tiêu chuẩn trong giải 8-puzzle vì vừa nhanh vừa đảm bảo tối ưu, đặc biệt khi h(n) tính toán đơn giản. Nhược điểm là A* vẫn có thể tiêu tốn nhiều bộ nhớ nếu không gian trạng thái lớn.

**•	IDA_Star:** Phiên bản IDDFS cho A*. IDA* thực hiện nhiều lần DFS có giới hạn dựa trên ngưỡng f=n=g+h, lần lặp sau tăng dần ngưỡng. Nhờ đó, IDA* chỉ sử dụng bộ nhớ O(d) nhỏ, nhưng phải lặp lại nhiều DFS và có thể lặp lại mở rộng một số nút nhiều lần. IDA* vẫn đảm bảo tìm lời giải ngắn nhất nếu heuristic đúng.

**Hình ảnh GIF minh họa:**
**- Greedy Best-First Search:**

![GBFS](https://github.com/user-attachments/assets/bbf9d104-a70f-4d92-a606-2316742145ce)

**- A_Star:**

![A_Star](https://github.com/user-attachments/assets/73c777a1-ff86-4556-ba6e-3d792e0fcb40)

**- IDA_Star:**

![IDA_Star](https://github.com/user-attachments/assets/2b0adf19-1600-4169-9000-78ed0ed1bb94)

**Hiệu suất:** A* thường vượt trội so với BFS/UCS nhờ heuristic giúp cắt giảm đáng kể số trạng thái thăm. Greedy rất nhanh nhưng đôi khi đi sai hướng, không đảm bảo tìm được hoặc giải đúng. IDA* tiết kiệm bộ nhớ hơn A* nhưng đổi lại tính toán nhiều. Nói chung, với bài 8-Puzzle kích thước nhỏ, A* (với heuristic Manhattan) thường được ưu tiên vì hiệu quả cao và lời giải tối ưu, trong khi Greedy/A* có thể thử nghiệm để so sánh tốc độ/độ chính xác.

**2.3 Tìm kiếm cục bộ (Hill Climbing, Simulated Annealing, Di truyền, Beam Search)**

Giới thiệu: Tìm kiếm cục bộ (Local Search) là nhóm thuật toán chỉ quan tâm đến trạng thái hiện tại và các trạng thái láng giềng, không xây dựng toàn bộ cây tìm kiếm. Mục tiêu của nhóm này là tối ưu hóa dần theo một hàm đánh giá (heuristic), thường là hàm phạt – ví dụ như khoảng cách Manhattan trong bài toán 8-Puzzle.

  **•	Trạng thái (state):** là một cấu hình cụ thể của bàn 8-Puzzle.

  **•	Mục tiêu (goal):** trạng thái hoàn chỉnh {1,2,3; 4,5,6; 7,8,0}.

  **•	Lời giải (solution):** là dãy các bước di chuyển (U/D/L/R) từ trạng thái ban đầu đến đích, thường là nghiệm xấp xỉ gần đúng (không đảm bảo tối ưu).

**• Hill Climbing (Leo đồi):** Hill Climbing là thuật toán cơ bản trong tìm kiếm cục bộ, hoạt động theo nguyên tắc: luôn di chuyển đến trạng thái kế tiếp có giá trị heuristic tốt hơn (thường là nhỏ hơn). Quá trình lặp lại cho đến khi không còn trạng thái kế nào tốt hơn – khi đó thuật toán dừng lại, có thể đã đạt đích hoặc mắc kẹt tại cực tiểu cục bộ.
Ưu điểm là đơn giản, nhanh, sử dụng rất ít bộ nhớ. Nhược điểm là dễ mắc bẫy cục bộ hoặc các vùng bằng phẳng (plateau) – nơi các trạng thái láng giềng có giá trị bằng nhau.
Thuật toán có 3 biến thể chính:

**•	1. Leo đồi đơn giản (Simple Hill Climbing):** Mỗi bước chỉ cần tìm được một trạng thái kế tiếp tốt hơn (giảm heuristic) và lập tức chuyển sang đó mà không so sánh tất cả các lựa chọn.
o	Ưu điểm: rất nhanh, dễ cài đặt.
o	Nhược điểm: dễ chọn nhầm hướng, bị kẹt sớm ở cực trị cục bộ do không chọn được hướng tốt nhất.

**•	2. Leo đồi dốc nhất (Steepest-Ascent Hill Climbing):** Ở mỗi bước, thuật toán duyệt tất cả các trạng thái kế tiếp hợp lệ và chọn ra trạng thái tốt nhất (giảm heuristic nhiều nhất).
o	Ưu điểm: giảm xác suất chọn sai hướng, dễ vượt qua vùng bằng phẳng.
o	Nhược điểm: tốn thời gian hơn vì cần kiểm tra toàn bộ nước đi.

**•	3. Leo đồi ngẫu nhiên (Stochastic Hill Climbing):** Không chọn trạng thái tốt nhất mà chọn ngẫu nhiên một trạng thái cải thiện được heuristic trong số các trạng thái láng giềng tốt hơn.
o	Ưu điểm: có thể tránh bị kẹt tại các điểm cục bộ, tăng khả năng tìm được nghiệm gần tối ưu.
o	Nhược điểm: không ổn định, chất lượng nghiệm phụ thuộc vào may rủi.

**• Simulated Annealing (Tôi luyện):** Là một cải tiến của Hill Climbing, cho phép đôi khi chấp nhận bước đi tệ hơn (heuristic tăng) với một xác suất nhất định. Xác suất này phụ thuộc vào một tham số gọi là “nhiệt độ” và giảm dần theo thời gian.
•	Khi nhiệt độ cao, thuật toán dễ dàng chấp nhận các bước tệ hơn, giúp thoát khỏi cực trị cục bộ.
•	Khi nhiệt độ hạ thấp, thuật toán tập trung tối ưu hướng tới lời giải.
Nhờ cơ chế này, SA có thể tiếp cận nghiệm toàn cục trong nhiều bài toán khó. Tuy nhiên, tốc độ chậm hơn và cần tinh chỉnh tham số như lịch làm nguội (cooling schedule), nhiệt độ ban đầu.

**• Thuật toán di truyền (Genetic Algorithm):** Mô phỏng quá trình tiến hóa tự nhiên. Mỗi lời giải (một dãy bước di chuyển) là một cá thể trong quần thể. Thuật toán thực hiện các vòng đời gồm:

**•	Chọn lọc (selection):** giữ lại các cá thể tốt (có fitness cao).

**•	Lai ghép (crossover):** kết hợp 2 cá thể để tạo cá thể mới.

**•	Đột biến (mutation):** thay đổi ngẫu nhiên một phần cá thể để tránh bị kẹt.

Fitness được đánh giá bằng khoảng cách từ trạng thái cuối đến mục tiêu.
Ưu điểm: khám phá không gian trạng thái rộng. Nhược điểm: không đảm bảo tối ưu, cần nhiều lần đánh giá, phù hợp hơn với bài toán lớn hoặc không yêu cầu chính xác.

**• Beam Search:** Biến thể của tìm kiếm theo chiều rộng có dùng heuristic. Thay vì mở rộng tất cả trạng thái ở mỗi bước, Beam Search chỉ giữ lại W trạng thái tốt nhất theo heuristic (ví dụ Manhattan thấp nhất).

•	Giảm mạnh bộ nhớ so với BFS vì chỉ lưu một số trạng thái chọn lọc.

•	Nếu beam width quá nhỏ, có thể bỏ lỡ lời giải tối ưu.

•	Beam Search được xem là sự cân bằng giữa BFS và Greedy: không quá "tham lam", cũng không mở rộng toàn bộ.

**Hình ảnh GIF minh họa:**
**- Simple Hill Climbing:**

![SimpleHC](https://github.com/user-attachments/assets/de6437e6-b212-49df-9c50-68ed57eb682d)

**- Steepest-Ascent Hill Climbing:**

![SteepestHC](https://github.com/user-attachments/assets/336f9759-af15-4d4d-b07b-ad990347f1ec)

**- Stochastic Hill Climbing:**

![StochasticHC](https://github.com/user-attachments/assets/7bd8a2d7-30eb-486c-a031-3f124a3296f2)

**- Simulated Annealing:**

![SimulatedAnnealing](https://github.com/user-attachments/assets/0e1173be-c9a3-4b24-bfda-b2c3523de8c4)

**- Genetic Algorithm:**

![Genitic](https://github.com/user-attachments/assets/b62479f0-ec0b-44b8-99e1-0c613789b1d1)

**- Beam Search:**

![Beam-Search](https://github.com/user-attachments/assets/fc1bb231-9d7e-4c6c-9d31-f44c9fe70939)

**Hiệu suất tổng quát của nhóm Local Search:**
**•	Hill Climbing (cả 3 biến thể):** chạy nhanh, rất tiết kiệm bộ nhớ.
o	Simple dễ bị kẹt.
o	Steepest đỡ kẹt hơn nhưng tốn tính toán.
o	Stochastic giảm xác suất kẹt, nhưng không ổn định.
•	Simulated Annealing: có khả năng vượt qua cực trị cục bộ tốt, tìm được nghiệm tốt nhưng chậm và khó điều chỉnh tham số.
•	Di truyền: mạnh khi không gian tìm kiếm lớn, phù hợp cho bài toán nhiều lời giải tiềm năng. Với 8-Puzzle thì hiệu quả kém hơn do không gian nhỏ, dễ bị kẹt ở nghiệm xấu.
•	Beam Search: chạy nhanh, dùng ít bộ nhớ, nhưng nếu beam nhỏ thì dễ sai hướng và không tìm được lời giải đúng.

**2.4 Môi trường phức tạp (And-Or Tree, Quan sát cục bộ, Môi trường động/không biết trước)**

**Giới thiệu:** Nhóm này giả lập các tình huống phức tạp như không chắc chắn hay thay đổi. 8-Puzzle vốn là môi trường quan sát hoàn chỉnh, song ta có thể mở rộng:
•	And-Or Search: Tìm kiếm trên cây And-Or (cây Đa/Nhỏ) thường dùng trong lập kế hoạch khi hành động có nhiều kết quả (ví dụ hành động “D” có thể thành công hay thất bại). Trong bối cảnh 8-Puzzle, ta có thể coi “không chắc chắn” là ngẫu nhiên: And-node đại diện cho hành động, Or-node đại diện cho trạng thái kết quả. Thuật toán tìm bản kế hoạch (chuỗi hành động) đảm bảo với mọi khả năng của môi trường. Ưu điểm: xử lý bài toán có nhánh phụ (AND) và lựa chọn (OR). Nhược: phức tạp lớn, ít được dùng thực tế cho 8-Puzzle.

**•	Quan sát cục bộ (Partially Observable):** Môi trường mà tác vụ của agent chỉ thu thập được thông tin không đầy đủ. Ví dụ 8-Puzzle quan sát cục bộ có thể giả định agent chỉ thấy một phần của bàn, cần duy trì tập tin tin (belief state) về các khả năng còn lại. Ở đây, thuật toán có thể chuyển tìm kiếm sang không gian tin: ví dụ dùng backtracking trên các biến khả năng, hoặc áp dụng POMDP (quy hoạch Markov quan sát cục bộ) để ước lượng. Tham khảo: POMDP và các chiến lược tiềm năng giúp agent quyết định ngay cả khi thông tin bị thiếu.

**•	Môi trường động/không biết trước:** Môi trường có thể thay đổi trong khi tìm kiếm (ví dụ ô trống chuyển vị trí ngẫu nhiên). Trong bài 8-Puzzle, một cách mô phỏng đơn giản là di chuyển vài bước ngẫu nhiên trước rồi mới tìm đường (thuật toán Online DFS-Agent của AIMA). Cách tiếp cận chung là “mở rộng tầm nhìn” trong khi chạy: cứ một khoảng thời gian agent lại tính toán lại đường đi với trạng thái mới. Như vậy, thuật toán kết hợp giữa khám phá (random walk) và tìm kiếm (DFS hoặc A* khi có đủ thông tin). Ứng dụng chính của nhóm này là trong các trò chơi/phương tiện động, còn với 8-Puzzle thuần túy thường dùng để minh hoạ khái niệm.

**Hình ảnh GIF minh họa:**
**- And-Or Search:**

![AndOrSearch](https://github.com/user-attachments/assets/32e1215f-baa8-4901-a857-e0b65e9db1f1)

**- Partially Observable:**

![PartialObsSearch](https://github.com/user-attachments/assets/a0522b0c-5f33-422b-8735-9ee69590fd95)

**- Unknown or Dynamic Environment:**

![UnknowDynamiSearch](https://github.com/user-attachments/assets/7e16b6a7-4aaf-4345-921b-8015217c4f4a)

**Hiệu suất:** Các phương pháp trong môi trường phức tạp nhằm giải quyết thiếu thông tin và động, đổi lại phức tạp tính toán rất cao. And-Or Search tạo ra kế hoạch chi tiết nhưng tăng nhanh theo số tình huống (And/Or) kết hợp. Môi trường quan sát cục bộ (POMDP) yêu cầu bảo toàn niềm tin, tăng rất nhiều trạng thái ảo. Tìm kiếm trong môi trường động/tự chỉ số đòi hỏi tính toán online liên tục, tốn thời gian. Đối với 8-Puzzle, đây chủ yếu là mở rộng khái niệm; hiệu suất thực tế phụ thuộc cài đặt, thường kém hơn các phương pháp thông tin đầy đủ do overhead quản lý tin và cập nhật.

**2.5 Tìm kiếm có ràng buộc (Backtracking, Forward Checking, AC-3)**

**Giới thiệu:** 8-Puzzle có thể xem như một CSP (Constraint Satisfaction Problem) với biến là mỗi vị trí trên bàn và ràng buộc AllDifferent (các ô 1–8 phải khác nhau). Các thuật toán tìm kiếm có ràng buộc khai thác tính chất này. Thành phần: biến (ô bàn), miền giá trị (số 0–8), ràng buộc (ví dụ AllDifferent, hay ràng buộc vị trí tương lai), mục tiêu là gán đủ các ô sao cho thỏa ràng buộc và đến đích.

**•	Backtracking:** Duyệt theo chiều sâu, gán giá trị cho các biến tuần tự. Nếu vi phạm ràng buộc thì quay lui (nhánh đó không tiếp tục). Theo AIMA, đây giống DFS có kiểm tra ràng buộc tại mỗi bước. Với 8-Puzzle, backtracking đơn giản chỉ gán lần lượt vị trí 0,1,…,8; do AC-3 (AllDifferent) gần như luôn có nghiệm, backtracking sẽ lần lượt sinh lời giải là đích. Về bản chất, backtracking chính là DFS trong không gian giá trị thay vì trạng thái.

**•	Forward Checking:** Mở rộng backtracking, sau mỗi bước gán biến, thuật toán loại bỏ khỏi miền các biến chưa gán những giá trị vi phạm ngay lập tức với ràng buộc hiện tại. Điều này giúp phát hiện mâu thuẫn sớm, giảm số nhánh cần dò. Ví dụ: nếu vừa đặt một con số tại ô (1,1), ta loại bỏ con số đó khỏi miền của các ô khác. Với 8-Puzzle, sau AC-3 (AllDifferent) các ô đã phần nào thu hẹp miền, forward checking hạn chế tìm kiếm.

**•	AC-3 (Arc Consistency):** Là thuật toán kiểm tra tính nhất quán cung cho CSP. Mục đích là đảm bảo với mỗi cặp biến liên quan (xi, xj), mọi giá trị còn trong miền xi có một giá trị tương thích ở xj. Thuật toán AC-3 lặp loại bỏ giá trị không thể thỏa ràng buộc cho tới khi mọi cung đều nhất quán hoặc miền bị rỗng. Trong 8-Puzzle, AC-3 với ràng buộc AllDifferent sẽ huỷ bỏ trạng thái không hợp lệ (ô lặp giá trị). Trong cài đặt, AC-3 xác nhận môi trường nhất quán với AllDifferent hoặc báo vô nghiệm. Kết quả của AC-3 có thể làm giảm miền biến đáng kể trước khi backtracking thực sự tìm lời giải.

**Hình ảnh GIF minh họa:**
**- AC-3 (Arc Consistency:**

![AC-3](https://github.com/user-attachments/assets/f0c4f133-48ec-4521-b367-c09f2d6493bc)

**- Backtracking:**

![BackTracKing](https://github.com/user-attachments/assets/550b1c61-1e05-45be-9b17-3e612620e606)

**- Forward Checking:**
  
![ForwardChecking](https://github.com/user-attachments/assets/3ca54f70-c421-451e-b9e7-014296cda312)

**Hiệu suất:** Các thuật toán CSP này thường áp dụng tốt cho bài toán có cấu trúc chặt chẽ (ví dụ Sudoku). Với 8-Puzzle, AC-3 rất nhanh để kiểm tra tính hợp lệ của trạng thái và forward checking giảm đáng kể các lựa chọn, vì vậy backtracking đạt hiệu năng cao hơn so với DFS thường. Tuy nhiên, do 8-Puzzle có nhiều trạng thái hợp lệ và giới hạn (chỉ 9 biến), phương pháp này cơ bản vẫn là một dạng DFS trong không gian gán biến và thường không nhanh bằng các tìm kiếm heuristic trong nhiệm vụ tìm đường đi. Nhóm này thể hiện tính học thuật về CSP hơn là phương án thực tế tối ưu cho 8-Puzzle.

**2.6 Học tăng cường (Q-Learning)**

**Giới thiệu:** Học tăng cường (Reinforcement Learning) cho 8-Puzzle đặt tác nhân học cách di chuyển ô đến mục tiêu qua tương tác nhiều lần. Cụ thể ở đây sử dụng Q-Learning – thuật toán RL không mô hình (model-free). Thành phần: trạng thái là cấu hình bàn, hành động là di chuyển L/R/U/D, mục tiêu là đạt trạng thái đích. Sau mỗi hành động, tác nhân nhận thưởng (reward) dương khi tiến gần mục tiêu hoặc thưởng 0 nếu chưa tới, và cập nhật bảng Q. Lời giải (policy) sau khi học xong là chuỗi hành động tối ưu từ trạng thái bắt đầu.

**•	Q-Learning:** Mỗi cặp (trạng thái, hành động) có giá trị Q thể hiện chất lượng của hành động tại trạng thái đó. Khi agent chạy thử (episodes), nó chọn hành động (ε-greedy) và cập nhật theo quy tắc:
Q(s,a) ← Q(s,a) + α [R + γ max<sub>a'</sub>Q(s',a') – Q(s,a)].
Qua nhiều episode, Q-Learning học dần chính sách tốt nhất. Với 8-Puzzle, reward có thể đặt dương khi đạt đích và 0 khi chưa. Cuối cùng thuật toán thu được một policy dẫn từ trạng thái đầu đến đích (hoặc gần đích).

**Hình ảnh GIF minh họa:**
**- Q-Learninge:**
  
![Q learning](https://github.com/user-attachments/assets/87aff73a-d110-488d-913b-09eaf6bcce4c)

**Hiệu suất:** Q-Learning mạnh trong các bài toán mà môi trường phức tạp, không biết trước, song với 8-Puzzle (không gian trạng thái nhỏ nhưng muộn nhận thưởng), nó thường yêu cầu rất nhiều episode huấn luyện để hội tụ. Lời giải Q-Learning không đảm bảo ngắn nhất và độ tin cậy (confidence) phụ thuộc việc học đủ. Tuy nhiên, nó minh hoạ cách agent tự học chính sách điều khiển. Tham khảo: Q-Learning tìm các hành động tối ưu qua tương tác và cập nhật Q-table. Đồ thị hiệu suất có thể so sánh số bước trung bình hoặc thời gian huấn luyện cần thiết để đạt độ tin cậy nhất định.

**2.7 Hình ảnh so sánh hiệu suất của các thuật toán.**
![image](https://github.com/user-attachments/assets/2cda9a0b-e4ff-4a50-aab0-29a4a8684f08)

__________
**3. Kết luận**

Đồ án đã thử nghiệm và so sánh nhiều nhóm thuật toán AI trên bài 8-Puzzle. Kết quả tổng quát cho thấy:
**•	Tìm kiếm không có thông tin:** BFS/UCS đảm bảo tìm đường đi ngắn nhất nhưng độ phức tạp rất cao cả về thời gian và bộ nhớ. DFS ít tốn bộ nhớ nhưng khó tìm lời giải tốt. IDS là giải pháp hòa giải tốt giữa độ tối ưu và bộ nhớ thấp.

**•	Tìm kiếm có thông tin:** A* (với heuristic Manhattan) thường vượt trội nhất, nhanh và tối ưu. Greedy tìm nhanh hơn nhưng đôi khi lạc hướng. IDA* tiết kiệm bộ nhớ nên nếu thiếu RAM thì có thể dùng thay A*.

**•	Tìm kiếm cục bộ:** Có tốc độ cao và tiết kiệm bộ nhớ, nhưng không đảm bảo tìm được lời giải ngắn nhất (hoặc thậm chí không tìm đến đích nếu bị kẹt). Hill Climbing đơn giản nhưng dễ sa lầy, Simulated Annealing có khả năng tìm được lời giải tốt hơn nhờ chấp nhận bước xấu. Di truyền/Beam Search là lựa chọn khi cần giảm sử dụng tài nguyên, nhưng cần hiệu chỉnh tham số.

**•	Môi trường phức tạp:** Các phương pháp này chủ yếu áp dụng cho tình huống bất định hoặc thay đổi, ít dùng trực tiếp cho 8-Puzzle cổ điển. Chúng cho thấy cách tiếp cận giải quyết môi trường quan sát cục bộ hoặc động, nhưng đổi lại gây tăng chi phí tính toán và phức tạp.

**•	Tìm kiếm có ràng buộc:** AC-3/Forward Checking kết hợp với backtracking có thể nhanh trong việc loại bỏ trạng thái không hợp lệ ban đầu, nhưng cuối cùng vẫn dựa trên DFS, thường không vượt trội so với tìm kiếm heuristic đối với 8-Puzzle.

**•	Học tăng cường:** Q-Learning cần thời gian huấn luyện lâu và lời giải không phải luôn tối ưu. Bù lại, nó cho thấy khả năng tự học chính sách qua nhiều lần tương tác.

Kết hợp đồ án cho phép so sánh toàn diện. Độ tin cậy của lời giải (solution confidence) được định nghĩa là xác suất thuật toán tìm đúng đích (1.0 nếu chắc chắn tìm được). Ví dụ: A* và UCS luôn đạt 1.0 nếu đủ bộ nhớ, trong khi Greedy/Q-Learning thường thấp hơn. Biểu đồ hiệu suất (số bước giải, thời gian chạy, confidence) có thể minh hoạ rằng A* thường dẫn đầu về cả tốc độ và độ tin cậy, còn các thuật toán cục bộ tuy nhanh nhưng hay thất bại, BFS/UCS tìm chính xác nhưng rất chậm, Q-Learning cần huấn luyện lâu mới cho kết quả khả quan.

Đối với bài 8-Puzzle (bài toán nhỏ, đích rõ ràng), ta khuyến nghị dùng A* (với heuristic Manhattan) vì đảm bảo tìm đường đi ngắn nhất và tốn ít bước tìm kiếm nhất trên hầu hết các trường hợp. Khi bộ nhớ hạn chế, có thể dùng IDA* thay thế. Nếu chỉ cần nghiệm “khá tốt” và lập tức (xuất kết quả nhanh), các thuật toán local search (Hill Climbing, SA) hoặc Beam Search có thể dùng nhưng phải cẩn thận vì có thể lạc hướng. Các nhóm khác (CSP, RL) phù hợp khi muốn khám phá khác biệt về mô hình hơn là tìm lời giải hiệu quả. Lời giải trình bày dưới dạng dãy các bước di chuyển (ví dụ “L, D, R, U, …”) từ trạng thái đầu đến trạng thái đích, đồng thời kết thúc khi đến được mục tiêu (hay khi hàm phạt bằng 0).

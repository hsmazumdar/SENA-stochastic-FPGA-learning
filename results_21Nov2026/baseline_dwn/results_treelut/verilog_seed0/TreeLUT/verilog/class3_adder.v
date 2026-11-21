module class3_adder(input wire clk, input wire [43:0] i, output wire [5:0] o);

reg[5:0] o_reg;
assign o = o_reg;

reg [3:0] stage0_adder0;
reg [3:0] stage0_adder1;
reg [2:0] stage0_adder2;
reg [2:0] stage0_adder3;
reg [2:0] stage0_adder4;
reg [2:0] stage0_adder5;
reg [2:0] stage0_adder6;
reg [1:0] stage0_adder7;
reg [1:0] stage0_adder8;
reg [1:0] stage0_adder9;
reg [1:0] stage0_adder10;
reg [1:0] stage0_adder11;
reg [1:0] stage0_adder12;
reg [1:0] stage0_adder13;
reg [1:0] stage0_adder14;
reg [0:0] stage0_adder15;

always @(*) begin
	stage0_adder0 = 1'd1 + i[2:0];
	stage0_adder1 = i[5:3] + i[7:6];
	stage0_adder2 = i[9:8] + i[11:10];
	stage0_adder3 = i[13:12] + i[15:14];
	stage0_adder4 = i[17:16] + i[19:18];
	stage0_adder5 = i[21:20] + i[23:22];
	stage0_adder6 = i[25:24] + i[26:26];
	stage0_adder7 = i[27:27] + i[28:28];
	stage0_adder8 = i[29:29] + i[30:30];
	stage0_adder9 = i[31:31] + i[32:32];
	stage0_adder10 = i[33:33] + i[34:34];
	stage0_adder11 = i[35:35] + i[36:36];
	stage0_adder12 = i[37:37] + i[38:38];
	stage0_adder13 = i[39:39] + i[40:40];
	stage0_adder14 = i[41:41] + i[42:42];
	stage0_adder15 = i[43:43];
end

reg [4:0] stage1_adder0;
reg [3:0] stage1_adder1;
reg [3:0] stage1_adder2;
reg [3:0] stage1_adder3;
reg [2:0] stage1_adder4;
reg [2:0] stage1_adder5;
reg [2:0] stage1_adder6;
reg [2:0] stage1_adder7;

always @(*) begin
	stage1_adder0 = stage0_adder0 + stage0_adder1;
	stage1_adder1 = stage0_adder2 + stage0_adder3;
	stage1_adder2 = stage0_adder4 + stage0_adder5;
	stage1_adder3 = stage0_adder6 + stage0_adder7;
	stage1_adder4 = stage0_adder8 + stage0_adder9;
	stage1_adder5 = stage0_adder10 + stage0_adder11;
	stage1_adder6 = stage0_adder12 + stage0_adder13;
	stage1_adder7 = stage0_adder14 + stage0_adder15;
end

reg [5:0] stage2_adder0;
reg [4:0] stage2_adder1;
reg [3:0] stage2_adder2;
reg [3:0] stage2_adder3;

always @(posedge clk) begin
	stage2_adder0 = stage1_adder0 + stage1_adder1;
	stage2_adder1 = stage1_adder2 + stage1_adder3;
	stage2_adder2 = stage1_adder4 + stage1_adder5;
	stage2_adder3 = stage1_adder6 + stage1_adder7;
end

reg [6:0] stage3_adder0;
reg [4:0] stage3_adder1;

always @(*) begin
	stage3_adder0 = stage2_adder0 + stage2_adder1;
	stage3_adder1 = stage2_adder2 + stage2_adder3;
end


always @(*) begin
o_reg = stage3_adder0 + stage3_adder1;
end

endmodule

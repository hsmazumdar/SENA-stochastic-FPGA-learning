module class7_tree22(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[48])) | ((~i[349]) & i[48]) | (i[338] & (~i[45]) & i[349] & i[48]) | ((~i[338]) & (~i[45]) & i[349] & i[48]) | (i[335] & i[98] & i[45] & i[349] & i[48]) | ((~i[335]) & i[98] & i[45] & i[349] & i[48]) | (i[121] & (~i[98]) & i[45] & i[349] & i[48]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
